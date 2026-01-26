#!/usr/bin/env python3
"""Compare GPTQ-quantized model with the original model.

This script compares:
- Model size and memory usage
- Inference speed
- Output quality (perplexity, generation similarity)
- Token-level agreement

Usage:
    python scripts/compare_models.py [--quantized-path PATH] [--original-model NAME]
    
Examples:
    # Compare default quantized model with original
    python scripts/compare_models.py
    
    # Compare with specific paths
    python scripts/compare_models.py --quantized-path outputs/my_quantized_model --original-model meta-llama/Llama-3.2-1B-Instruct
"""

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm


@dataclass
class ModelMetrics:
    """Container for model metrics."""
    name: str
    num_parameters: int
    model_size_mb: float
    gpu_memory_mb: float
    avg_inference_time: float
    tokens_per_second: float
    perplexity: Optional[float] = None


def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def count_parameters(model) -> int:
    """Count total model parameters."""
    return sum(p.numel() for p in model.parameters())


def get_model_size_mb(model) -> float:
    """Get model size in MB."""
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024


def load_original_model(model_name: str, device: str = "cuda:0"):
    """Load the original (non-quantized) model."""
    print(f"\n{'='*60}")
    print(f"Loading original model: {model_name}")
    print(f"{'='*60}")
    
    # Add project root to path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    
    from transformers import AutoTokenizer
    from analog_ptq.models.loader import load_model
    
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    
    initial_memory = get_gpu_memory_mb()
    
    model = load_model(
        model_name,
        dtype="float16",
        device_map=device,
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    final_memory = get_gpu_memory_mb()
    
    print(f"  Parameters: {count_parameters(model):,}")
    print(f"  Model size: {get_model_size_mb(model):.2f} MB")
    print(f"  GPU memory used: {final_memory - initial_memory:.2f} MB")
    
    return model, tokenizer


def load_quantized_model(model_path: str, device: str = "cuda:0"):
    """Load the quantized model."""
    print(f"\n{'='*60}")
    print(f"Loading quantized model: {model_path}")
    print(f"{'='*60}")
    
    # Add project root to path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    
    from transformers import AutoTokenizer
    from analog_ptq.models.loader import load_model
    from analog_ptq.quantization.utils import QuantizedLinear
    
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    
    # Load quantization config
    config_path = os.path.join(model_path, "quantization_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            quant_config = json.load(f)
        original_model = quant_config.get("original_model", "meta-llama/Llama-3.2-1B-Instruct")
        bits = quant_config.get("bits", 4)
        group_size = quant_config.get("group_size", 128)
    else:
        original_model = "meta-llama/Llama-3.2-1B-Instruct"
        bits = 4
        group_size = 128
    
    tokenizer = AutoTokenizer.from_pretrained(original_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    initial_memory = get_gpu_memory_mb()
    
    # Load base model
    model = load_model(
        original_model,
        dtype="float16",
        device_map=device,
    )
    
    # Load quantized weights
    state_dict_path = os.path.join(model_path, "model.safetensors")
    if not os.path.exists(state_dict_path):
        state_dict_path = os.path.join(model_path, "pytorch_model.bin")
    
    if os.path.exists(state_dict_path):
        if state_dict_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(state_dict_path)
        else:
            state_dict = torch.load(state_dict_path, map_location="cpu")
        
        # Find and restore quantized layers
        quantized_layers = set()
        for key in state_dict.keys():
            if ".qweight" in key:
                layer_path = key.rsplit(".qweight", 1)[0]
                quantized_layers.add(layer_path)
        
        for layer_path in quantized_layers:
            parts = layer_path.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            
            layer_name = parts[-1]
            original_layer = getattr(parent, layer_name)
            
            qlayer = QuantizedLinear(
                in_features=original_layer.in_features,
                out_features=original_layer.out_features,
                bits=bits,
                group_size=group_size,
                bias=original_layer.bias is not None,
            )
            
            qlayer.qweight.copy_(state_dict[f"{layer_path}.qweight"])
            qlayer.scales.copy_(state_dict[f"{layer_path}.scales"])
            qlayer.zeros.copy_(state_dict[f"{layer_path}.zeros"])
            if original_layer.bias is not None and f"{layer_path}.bias" in state_dict:
                qlayer.bias.copy_(state_dict[f"{layer_path}.bias"])
            
            setattr(parent, layer_name, qlayer.to(model.device))
    
    model.eval()
    final_memory = get_gpu_memory_mb()
    
    print(f"  Parameters: {count_parameters(model):,}")
    print(f"  Model size: {get_model_size_mb(model):.2f} MB")
    print(f"  GPU memory used: {final_memory - initial_memory:.2f} MB")
    print(f"  Quantized layers: {len(quantized_layers)}")
    
    return model, tokenizer


def measure_inference_speed(model, tokenizer, num_runs: int = 5) -> Tuple[float, float]:
    """Measure inference speed."""
    prompts = [
        "The quick brown fox jumps over",
        "In the beginning, there was",
        "Machine learning is a field of",
        "The capital of France is",
        "To be or not to be,",
    ]
    
    device = next(model.parameters()).device
    total_time = 0
    total_tokens = 0
    
    # Warmup
    inputs = tokenizer(prompts[0], return_tensors="pt").to(device)
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    
    # Actual runs
    for _ in range(num_runs):
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=30,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            elapsed = time.time() - start_time
            
            new_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
            total_time += elapsed
            total_tokens += new_tokens
    
    avg_time = total_time / (num_runs * len(prompts))
    tokens_per_sec = total_tokens / total_time
    
    return avg_time, tokens_per_sec


def compute_perplexity(model, tokenizer, texts: List[str]) -> float:
    """Compute perplexity on a set of texts."""
    device = next(model.parameters()).device
    total_loss = 0
    total_tokens = 0
    
    for text in tqdm(texts, desc="Computing perplexity"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            
            total_loss += loss.item() * inputs["input_ids"].shape[1]
            total_tokens += inputs["input_ids"].shape[1]
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return perplexity


def compare_outputs(original_model, quantized_model, tokenizer, prompts: List[str]) -> Dict:
    """Compare outputs between original and quantized models."""
    device = next(original_model.parameters()).device
    
    results = {
        "prompts": [],
        "logit_cosine_similarity": [],
        "top1_agreement": [],
        "top5_agreement": [],
        "kl_divergence": [],
    }
    
    print("\n" + "="*60)
    print("Comparing model outputs...")
    print("="*60)
    
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            orig_outputs = original_model(**inputs)
            quant_outputs = quantized_model(**inputs)
            
            orig_logits = orig_outputs.logits[:, -1, :].float()
            quant_logits = quant_outputs.logits[:, -1, :].float()
            
            # Cosine similarity
            cosine_sim = F.cosine_similarity(orig_logits, quant_logits, dim=-1).mean().item()
            
            # Top-1 agreement
            orig_top1 = orig_logits.argmax(dim=-1)
            quant_top1 = quant_logits.argmax(dim=-1)
            top1_agree = (orig_top1 == quant_top1).float().mean().item()
            
            # Top-5 agreement
            orig_top5 = orig_logits.topk(5, dim=-1).indices
            quant_top5 = quant_logits.topk(5, dim=-1).indices
            top5_agree = sum(1 for t in quant_top5[0] if t in orig_top5[0]) / 5
            
            # KL divergence
            orig_probs = F.softmax(orig_logits, dim=-1)
            quant_probs = F.softmax(quant_logits, dim=-1)
            kl_div = F.kl_div(quant_probs.log(), orig_probs, reduction='batchmean').item()
            
            results["prompts"].append(prompt)
            results["logit_cosine_similarity"].append(cosine_sim)
            results["top1_agreement"].append(top1_agree)
            results["top5_agreement"].append(top5_agree)
            results["kl_divergence"].append(kl_div)
            
            # Print comparison
            orig_token = tokenizer.decode([orig_top1.item()])
            quant_token = tokenizer.decode([quant_top1.item()])
            match = "✓" if orig_token == quant_token else "✗"
            
            print(f"\nPrompt: {prompt!r}")
            print(f"  Original next token: {orig_token!r}")
            print(f"  Quantized next token: {quant_token!r} {match}")
            print(f"  Cosine similarity: {cosine_sim:.4f}")
            print(f"  Top-5 overlap: {top5_agree:.1%}")
    
    return results


def generate_comparison_samples(original_model, quantized_model, tokenizer, prompts: List[str]):
    """Generate text samples from both models for comparison."""
    device = next(original_model.parameters()).device
    
    print("\n" + "="*60)
    print("Generation comparison...")
    print("="*60)
    
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            orig_output = original_model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            quant_output = quantized_model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        orig_text = tokenizer.decode(orig_output[0], skip_special_tokens=True)
        quant_text = tokenizer.decode(quant_output[0], skip_special_tokens=True)
        
        print(f"\nPrompt: {prompt!r}")
        print(f"Original:  {orig_text}")
        print(f"Quantized: {quant_text}")


def main():
    parser = argparse.ArgumentParser(description="Compare quantized model with original")
    parser.add_argument(
        "--quantized-path",
        type=str,
        default="outputs/llama3.2-1b-instruct-gptq/quantized_model",
        help="Path to quantized model",
    )
    parser.add_argument(
        "--original-model",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Original model name or path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use",
    )
    parser.add_argument(
        "--skip-perplexity",
        action="store_true",
        help="Skip perplexity computation (faster)",
    )
    args = parser.parse_args()
    
    # Check paths
    if not os.path.exists(args.quantized_path):
        print(f"Error: Quantized model not found at {args.quantized_path}")
        return 1
    
    # Test prompts
    prompts = [
        "The capital of France is",
        "In machine learning, a neural network",
        "The quick brown fox jumps",
        "To solve this problem, we need to",
        "The meaning of life is",
    ]
    
    # Sample texts for perplexity
    perplexity_texts = [
        "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.",
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "In the year 2024, technology continues to advance at an unprecedented rate.",
        "The capital cities of Europe include Paris, London, Berlin, and Rome.",
        "Python is a popular programming language known for its simplicity and readability.",
    ]
    
    # Load models
    original_model, tokenizer = load_original_model(args.original_model, args.device)
    
    # Measure original model
    print("\nMeasuring original model performance...")
    orig_avg_time, orig_tps = measure_inference_speed(original_model, tokenizer)
    
    orig_metrics = ModelMetrics(
        name="Original",
        num_parameters=count_parameters(original_model),
        model_size_mb=get_model_size_mb(original_model),
        gpu_memory_mb=get_gpu_memory_mb(),
        avg_inference_time=orig_avg_time,
        tokens_per_second=orig_tps,
    )
    
    if not args.skip_perplexity:
        print("Computing perplexity for original model...")
        orig_metrics.perplexity = compute_perplexity(original_model, tokenizer, perplexity_texts)
    
    # Clear memory and load quantized model
    del original_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    
    quantized_model, _ = load_quantized_model(args.quantized_path, args.device)
    
    # Measure quantized model
    print("\nMeasuring quantized model performance...")
    quant_avg_time, quant_tps = measure_inference_speed(quantized_model, tokenizer)
    
    quant_metrics = ModelMetrics(
        name="Quantized",
        num_parameters=count_parameters(quantized_model),
        model_size_mb=get_model_size_mb(quantized_model),
        gpu_memory_mb=get_gpu_memory_mb(),
        avg_inference_time=quant_avg_time,
        tokens_per_second=quant_tps,
    )
    
    if not args.skip_perplexity:
        print("Computing perplexity for quantized model...")
        quant_metrics.perplexity = compute_perplexity(quantized_model, tokenizer, perplexity_texts)
    
    # Reload original for comparison
    original_model, _ = load_original_model(args.original_model, args.device)
    
    # Compare outputs
    comparison_results = compare_outputs(original_model, quantized_model, tokenizer, prompts)
    
    # Generate samples
    generate_comparison_samples(original_model, quantized_model, tokenizer, prompts[:3])
    
    # Print summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    print(f"\n{'Metric':<30} {'Original':>15} {'Quantized':>15} {'Change':>15}")
    print("-" * 75)
    
    print(f"{'Parameters':<30} {orig_metrics.num_parameters:>15,} {quant_metrics.num_parameters:>15,} "
          f"{(quant_metrics.num_parameters/orig_metrics.num_parameters - 1)*100:>14.1f}%")
    
    print(f"{'Model Size (MB)':<30} {orig_metrics.model_size_mb:>15.2f} {quant_metrics.model_size_mb:>15.2f} "
          f"{(quant_metrics.model_size_mb/orig_metrics.model_size_mb - 1)*100:>14.1f}%")
    
    print(f"{'Avg Inference Time (s)':<30} {orig_metrics.avg_inference_time:>15.3f} {quant_metrics.avg_inference_time:>15.3f} "
          f"{(quant_metrics.avg_inference_time/orig_metrics.avg_inference_time - 1)*100:>14.1f}%")
    
    print(f"{'Tokens/Second':<30} {orig_metrics.tokens_per_second:>15.1f} {quant_metrics.tokens_per_second:>15.1f} "
          f"{(quant_metrics.tokens_per_second/orig_metrics.tokens_per_second - 1)*100:>14.1f}%")
    
    if orig_metrics.perplexity and quant_metrics.perplexity:
        print(f"{'Perplexity':<30} {orig_metrics.perplexity:>15.2f} {quant_metrics.perplexity:>15.2f} "
              f"{(quant_metrics.perplexity/orig_metrics.perplexity - 1)*100:>14.1f}%")
    
    # Output quality metrics
    avg_cosine = sum(comparison_results["logit_cosine_similarity"]) / len(comparison_results["logit_cosine_similarity"])
    avg_top1 = sum(comparison_results["top1_agreement"]) / len(comparison_results["top1_agreement"])
    avg_top5 = sum(comparison_results["top5_agreement"]) / len(comparison_results["top5_agreement"])
    avg_kl = sum(comparison_results["kl_divergence"]) / len(comparison_results["kl_divergence"])
    
    print("\n" + "-" * 75)
    print(f"{'Avg Logit Cosine Similarity':<30} {avg_cosine:>15.4f}")
    print(f"{'Avg Top-1 Agreement':<30} {avg_top1:>15.1%}")
    print(f"{'Avg Top-5 Agreement':<30} {avg_top5:>15.1%}")
    print(f"{'Avg KL Divergence':<30} {avg_kl:>15.4f}")
    
    print("\n" + "="*60)
    print("✅ Comparison complete!")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

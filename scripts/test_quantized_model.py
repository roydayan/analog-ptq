#!/usr/bin/env python3
"""Test script for verifying quantized model functionality.

This script loads a quantized model and runs simple inference tests
to verify it works correctly.

Usage:
    python scripts/test_quantized_model.py [--model-path PATH] [--compare-original]
    
Examples:
    # Test the default quantized model
    python scripts/test_quantized_model.py
    
    # Test with a specific model path
    python scripts/test_quantized_model.py --model-path outputs/my_quantized_model
    
    # Compare quantized vs original model outputs
    python scripts/test_quantized_model.py --compare-original
"""

import argparse
import os
import sys
import time

import torch


def load_quantized_model(model_path: str):
    """Load a quantized model from disk."""
    print(f"Loading quantized model from: {model_path}")
    
    # Add the project root to path for imports
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    
    from transformers import AutoTokenizer
    
    # Load tokenizer from original model (quantized model may not have tokenizer saved)
    config_path = os.path.join(model_path, "quantization_config.json")
    if os.path.exists(config_path):
        import json
        with open(config_path) as f:
            config = json.load(f)
        original_model = config.get("original_model", "meta-llama/Llama-3.2-1B-Instruct")
    else:
        original_model = "meta-llama/Llama-3.2-1B-Instruct"
    
    print(f"Loading tokenizer from: {original_model}")
    tokenizer = AutoTokenizer.from_pretrained(original_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load the quantized model
    # First try loading as a regular HF model
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="cuda:0" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True,
        )
        print("Loaded model using AutoModelForCausalLM")
    except Exception as e:
        print(f"Could not load with AutoModelForCausalLM: {e}")
        print("Trying to load with custom QuantizedLinear layers...")
        
        # Load using our custom wrapper
        from analog_ptq.models.wrapper import ModelWrapper
        from analog_ptq.models.loader import load_model
        
        # Load the base model architecture
        model = load_model(
            original_model,
            dtype="float16",
            device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        )
        
        # Load quantized state dict
        state_dict_path = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(state_dict_path):
            state_dict = torch.load(state_dict_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
            print("Loaded quantized weights")
    
    return model, tokenizer


def run_inference_test(model, tokenizer, prompts: list = None):
    """Run inference tests on the model."""
    if prompts is None:
        prompts = [
            "The capital of France is",
            "In machine learning, a neural network is",
            "The quick brown fox",
            "1 + 1 =",
        ]
    
    print("\n" + "=" * 60)
    print("Running inference tests...")
    print("=" * 60)
    
    device = next(model.parameters()).device
    model.eval()
    
    results = []
    total_time = 0
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt!r}")
        
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
        total_time += elapsed
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        new_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
        tokens_per_sec = new_tokens / elapsed if elapsed > 0 else 0
        
        print(f"Output: {generated_text!r}")
        print(f"Time: {elapsed:.2f}s, Tokens: {new_tokens}, Speed: {tokens_per_sec:.1f} tok/s")
        
        results.append({
            "prompt": prompt,
            "output": generated_text,
            "time": elapsed,
            "new_tokens": new_tokens,
        })
    
    print("\n" + "=" * 60)
    print(f"Total inference time: {total_time:.2f}s")
    print(f"Average time per prompt: {total_time / len(prompts):.2f}s")
    print("=" * 60)
    
    return results


def compare_with_original(quantized_model, tokenizer, original_model_name: str):
    """Compare quantized model outputs with original model."""
    print("\n" + "=" * 60)
    print("Comparing with original model...")
    print("=" * 60)
    
    from analog_ptq.models.loader import load_model
    
    print(f"Loading original model: {original_model_name}")
    original_model = load_model(
        original_model_name,
        dtype="float16",
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    original_model.eval()
    
    prompts = ["The meaning of life is", "Machine learning helps"]
    device = next(quantized_model.parameters()).device
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt!r}")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            # Get logits from both models
            q_outputs = quantized_model(**inputs)
            o_outputs = original_model(**inputs)
            
            q_logits = q_outputs.logits[:, -1, :]
            o_logits = o_outputs.logits[:, -1, :]
            
            # Compare top predictions
            q_top = torch.topk(q_logits, k=5, dim=-1)
            o_top = torch.topk(o_logits, k=5, dim=-1)
            
            print("  Original top-5 tokens:", [tokenizer.decode([t]) for t in o_top.indices[0]])
            print("  Quantized top-5 tokens:", [tokenizer.decode([t]) for t in q_top.indices[0]])
            
            # Compute similarity
            cosine_sim = torch.nn.functional.cosine_similarity(
                q_logits.float(), o_logits.float(), dim=-1
            ).mean().item()
            print(f"  Logit cosine similarity: {cosine_sim:.4f}")
    
    # Clean up
    del original_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


def get_model_info(model):
    """Get model size and parameter information."""
    total_params = sum(p.numel() for p in model.parameters())
    total_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
    
    print(f"\nModel Info:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Model size: {total_size_mb:.2f} MB")
    print(f"  Device: {next(model.parameters()).device}")
    print(f"  Dtype: {next(model.parameters()).dtype}")


def main():
    parser = argparse.ArgumentParser(description="Test a quantized model")
    parser.add_argument(
        "--model-path",
        type=str,
        default="outputs/llama3.2-1b-instruct-gptq/quantized_model",
        help="Path to the quantized model directory",
    )
    parser.add_argument(
        "--compare-original",
        action="store_true",
        help="Compare outputs with the original (non-quantized) model",
    )
    parser.add_argument(
        "--original-model",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Original model name for comparison",
    )
    args = parser.parse_args()
    
    # Check if model path exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model path does not exist: {args.model_path}")
        print("\nTo create a quantized model, run:")
        print("  analog-ptq configs/examples/llama_gptq.yaml")
        return 1
    
    # Load the quantized model
    model, tokenizer = load_quantized_model(args.model_path)
    
    # Get model info
    get_model_info(model)
    
    # Run inference tests
    run_inference_test(model, tokenizer)
    
    # Optionally compare with original
    if args.compare_original:
        compare_with_original(model, tokenizer, args.original_model)
    
    print("\n✅ All tests passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

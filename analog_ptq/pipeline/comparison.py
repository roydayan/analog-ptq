"""Comparison runner for comparing multiple model variants (Original, GPTQ, NA-GPTQ).

This module provides the ComparisonRunner class that orchestrates comparing
multiple quantization methods with optional noise injection across benchmarks.

Example usage:
    >>> config = load_comparison_config("configs/comparison.yaml")
    >>> runner = ComparisonRunner(config)
    >>> results = runner.run()
"""

import copy
import gc
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from analog_ptq.config.schema import (
    ComparisonExperimentConfig,
    ModelVariantConfig,
    load_config,
)
from analog_ptq.evaluation.harness import LMEvalHarness
from analog_ptq.evaluation.metrics import MetricsCollector
from analog_ptq.models.loader import load_model, load_tokenizer
from analog_ptq.models.wrapper import ModelWrapper
from analog_ptq.noise import WeightNoiseInjector, DynamicNoiseWrapper
from analog_ptq.pipeline.registry import registry
from analog_ptq.quantization.base import check_cached_model, load_cached_quantized_model
from analog_ptq.quantization.calibration import CalibrationDataLoader
from analog_ptq.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class VariantResult:
    """Results for a single model variant."""
    
    name: str
    quantization_method: Optional[str] = None
    noise_enabled: bool = False
    noise_function: Optional[str] = None
    
    # Model metrics
    num_parameters: int = 0
    model_size_mb: float = 0.0
    gpu_memory_mb: float = 0.0
    
    # Quantization metrics
    num_quantized_layers: int = 0
    weights_in_quantized_layers: int = 0
    quantization_coverage_percent: float = 0.0
    
    # Evaluation results
    eval_results: Dict[str, Any] = field(default_factory=dict)
    
    # Timing
    quantization_time_seconds: float = 0.0
    evaluation_time_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "quantization_method": self.quantization_method,
            "noise_enabled": self.noise_enabled,
            "noise_function": self.noise_function,
            "num_parameters": self.num_parameters,
            "model_size_mb": self.model_size_mb,
            "gpu_memory_mb": self.gpu_memory_mb,
            "num_quantized_layers": self.num_quantized_layers,
            "weights_in_quantized_layers": self.weights_in_quantized_layers,
            "quantization_coverage_percent": self.quantization_coverage_percent,
            "eval_results": self.eval_results,
            "quantization_time_seconds": self.quantization_time_seconds,
            "evaluation_time_seconds": self.evaluation_time_seconds,
        }


class ComparisonRunner:
    """Orchestrates comparing multiple model variants.
    
    Handles:
    - Loading base model
    - Creating and quantizing variants (GPTQ, NA-GPTQ, etc.)
    - Applying noise to variants
    - Evaluating all variants on benchmarks
    - Generating comparison reports
    
    Example:
        >>> config = ComparisonExperimentConfig.from_yaml("configs/comparison.yaml")
        >>> runner = ComparisonRunner(config)
        >>> results = runner.run()
    """
    
    def __init__(self, config: ComparisonExperimentConfig):
        """Initialize the comparison runner.
        
        Args:
            config: Comparison experiment configuration
        """
        self.config = config
        self.output_dir = Path(config.experiment.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set random seed
        self._set_seed(config.experiment.seed)
        
        # Shared resources
        self.base_model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.calibration_data: Optional[List] = None
        
        # Results
        self.results: List[VariantResult] = []
    
    def _set_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        import random
        import numpy as np
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def _clear_gpu_memory(self) -> None:
        """Clear GPU memory between variants."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def _load_base_model(self) -> PreTrainedModel:
        """Load the base model (called once, cloned for each variant)."""
        logger.info("=" * 60)
        logger.info("Loading base model...")
        logger.info("=" * 60)
        
        model_config = self.config.model
        
        self.base_model = load_model(
            name_or_path=model_config.name_or_path,
            dtype=model_config.dtype,
            device_map=model_config.device_map,
            trust_remote_code=model_config.trust_remote_code,
            revision=model_config.revision,
        )
        
        self.tokenizer = load_tokenizer(
            name_or_path=model_config.name_or_path,
            trust_remote_code=model_config.trust_remote_code,
        )
        
        logger.info(f"  Model: {model_config.name_or_path}")
        logger.info(f"  Parameters: {sum(p.numel() for p in self.base_model.parameters()):,}")
        
        return self.base_model
    
    def _prepare_calibration_data(self) -> List:
        """Prepare calibration data (shared across all quantized variants)."""
        if self.calibration_data is not None:
            return self.calibration_data
        
        logger.info("=" * 60)
        logger.info("Preparing calibration data...")
        logger.info("=" * 60)
        
        calib_config = self.config.comparison.calibration
        
        loader = CalibrationDataLoader(
            tokenizer=self.tokenizer,
            seq_length=calib_config.seq_length,
        )
        
        self.calibration_data = loader.load_dataset_samples(
            dataset_name=calib_config.dataset,
            num_samples=calib_config.num_samples,
            seed=calib_config.seed,
        )
        
        logger.info(f"  Loaded {len(self.calibration_data)} calibration samples")
        
        return self.calibration_data
    
    def _create_fresh_model(self, needs_quantization: bool = False) -> ModelWrapper:
        """Create a fresh copy of the model for a new variant.
        
        This reloads the model to ensure clean state for each variant.
        
        Args:
            needs_quantization: If True, force single-device loading to avoid
                               multi-GPU tensor device mismatches during quantization.
        """
        self._clear_gpu_memory()
        
        model_config = self.config.model
        
        # For quantization, we need all tensors on the same device
        # device_map="auto" can spread across multiple GPUs which causes issues
        device_map = model_config.device_map
        if needs_quantization and device_map == "auto":
            device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
            logger.info(f"  Using single device {device_map} for quantization (overriding 'auto')")
        
        model = load_model(
            name_or_path=model_config.name_or_path,
            dtype=model_config.dtype,
            device_map=device_map,
            trust_remote_code=model_config.trust_remote_code,
            revision=model_config.revision,
        )
        
        return ModelWrapper(model)
    
    def _process_variant(self, variant: ModelVariantConfig) -> VariantResult:
        """Process a single model variant.
        
        Args:
            variant: Configuration for this variant
            
        Returns:
            Results for this variant
        """
        logger.info("=" * 60)
        logger.info(f"Processing variant: {variant.name}")
        logger.info("=" * 60)
        
        result = VariantResult(name=variant.name)
        metrics = MetricsCollector()
        
        # Check for cached quantized model
        variant_output = self.output_dir / f"{variant.name}_model"
        force_requantize = self.config.comparison.force_requantize
        
        # Try to load cached model if:
        # 1. Quantization is configured
        # 2. Cached model exists
        # 3. force_requantize is False
        if (variant.quantization is not None 
            and check_cached_model(variant_output) 
            and not force_requantize):
            
            logger.info(f"  Found cached model at {variant_output}")
            logger.info(f"  Loading cached model (use force_requantize: true to re-quantize)")
            
            quant_config = variant.quantization
            result.quantization_method = quant_config.method
            
            # Determine device_map for loading
            model_config = self.config.model
            device_map = model_config.device_map
            if device_map == "auto":
                device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
            
            # Load cached model
            try:
                model, quant_config_dict = load_cached_quantized_model(
                    variant_output,
                    device_map=device_map,
                    dtype=model_config.dtype,
                )
                wrapper = ModelWrapper(model)
                
                # Measure metrics from loaded model
                metrics.measure_model_size(wrapper.model)
                result.num_parameters = sum(p.numel() for p in wrapper.model.parameters())
                metrics.count_quantized_parameters(wrapper)
                perf_metrics = metrics.get_metrics()
                result.num_quantized_layers = perf_metrics.num_quantized_layers
                result.weights_in_quantized_layers = perf_metrics.quantized_weights
                if perf_metrics.original_params > 0:
                    result.quantization_coverage_percent = (
                        perf_metrics.quantized_weights / perf_metrics.original_params * 100
                    )
                result.quantization_time_seconds = 0.0  # Cached, no time spent
                
                logger.info(f"  Loaded cached model successfully")
            except Exception as e:
                logger.warning(f"  Failed to load cached model: {e}")
                logger.info(f"  Falling back to re-quantization...")
                # Fall through to regular quantization
                wrapper = None
        else:
            wrapper = None
        
        # If no cached model was loaded, create fresh model and quantize
        if wrapper is None:
            # Create fresh model (use single device if quantization is needed)
            needs_quantization = variant.quantization is not None
            wrapper = self._create_fresh_model(needs_quantization=needs_quantization)
            
            # Get baseline metrics
            metrics.measure_model_size(wrapper.model)
            result.num_parameters = sum(p.numel() for p in wrapper.model.parameters())
            
            # Apply quantization if configured
            if variant.quantization is not None:
                quant_start = datetime.now()
                
                quant_config = variant.quantization
                result.quantization_method = quant_config.method
                
                logger.info(f"  Applying {quant_config.method} quantization...")
                logger.info(f"    bits={quant_config.bits}, group_size={quant_config.group_size}")
                
                # Get quantizer
                quantizer_cls = registry.get_quantizer(quant_config.method)
                quantizer = quantizer_cls(
                    bits=quant_config.bits,
                    group_size=quant_config.group_size,
                    symmetric=quant_config.symmetric,
                    damp_percent=quant_config.damp_percent,
                    block_size=quant_config.block_size,
                    actorder=quant_config.actorder,
                    **quant_config.extra,
                )
                
                # Prepare calibration data
                calibration_data = self._prepare_calibration_data()
                
                # Quantize
                wrapper = quantizer.prepare(wrapper)
                wrapper = quantizer.quantize(wrapper, calibration_data)
                
                quant_end = datetime.now()
                result.quantization_time_seconds = (quant_end - quant_start).total_seconds()
                
                # Measure post-quantization metrics
                metrics.count_quantized_parameters(wrapper)
                perf_metrics = metrics.get_metrics()
                result.num_quantized_layers = perf_metrics.num_quantized_layers
                result.weights_in_quantized_layers = perf_metrics.quantized_weights
                # Compute coverage: quantized weights / original params
                if perf_metrics.original_params > 0:
                    result.quantization_coverage_percent = (
                        perf_metrics.quantized_weights / perf_metrics.original_params * 100
                    )
                else:
                    result.quantization_coverage_percent = 0.0
                
                # Save quantized model
                quantizer.save(wrapper, variant_output, original_model=self.config.model.name_or_path)
                logger.info(f"  Saved quantized model to {variant_output}")
        
        # Apply noise if configured
        if variant.noise is not None and variant.noise.enabled:
            noise_config = variant.noise
            result.noise_enabled = True
            result.noise_function = noise_config.function
            
            logger.info(f"  Applying noise: {noise_config.function}")
            logger.info(f"    params={noise_config.function_params}, mode={noise_config.mode}")
            
            if noise_config.mode in ("static", "both"):
                injector = WeightNoiseInjector(
                    noise_function=noise_config.function,
                    function_params=noise_config.function_params,
                    seed=noise_config.seed,
                    target_layers=noise_config.target_layers,
                )
                wrapper = injector.apply(wrapper)
            
            if noise_config.mode in ("dynamic", "both"):
                dynamic_wrapper = DynamicNoiseWrapper(
                    noise_function=noise_config.function,
                    function_params=noise_config.function_params,
                    target_layers=noise_config.target_layers,
                )
                wrapper = dynamic_wrapper.apply(wrapper)
        
        # Measure final model size
        result.model_size_mb = sum(
            p.numel() * p.element_size() for p in wrapper.model.parameters()
        ) / 1024 / 1024
        
        if torch.cuda.is_available():
            result.gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
        
        # Evaluate if configured
        if self.config.evaluation is not None:
            eval_start = datetime.now()
            
            eval_config = self.config.evaluation
            logger.info(f"  Evaluating on tasks: {eval_config.tasks}")
            
            harness = LMEvalHarness(
                model=wrapper,
                tokenizer=self.tokenizer,
                device=eval_config.device,
            )
            
            eval_results = harness.evaluate(
                tasks=eval_config.tasks,
                num_fewshot=eval_config.num_fewshot,
                batch_size=eval_config.batch_size,
                limit=eval_config.limit,
            )
            
            result.eval_results = eval_results.get("results", {})
            
            eval_end = datetime.now()
            result.evaluation_time_seconds = (eval_end - eval_start).total_seconds()
        
        # Clean up
        del wrapper
        self._clear_gpu_memory()
        
        return result
    
    def run(self) -> Dict[str, Any]:
        """Run the full comparison experiment.
        
        Returns:
            Dictionary containing comparison results
        """
        logger.info("=" * 60)
        logger.info(f"Starting comparison experiment: {self.config.experiment.name}")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        
        # Load base model and tokenizer (for tokenizer and calibration)
        self._load_base_model()
        
        # Process each enabled variant
        enabled_variants = self.config.comparison.get_enabled_variants()
        logger.info(f"Processing {len(enabled_variants)} variants...")
        
        for i, variant in enumerate(enabled_variants):
            logger.info(f"\n[{i+1}/{len(enabled_variants)}] {variant.name}")
            result = self._process_variant(variant)
            self.results.append(result)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Build final results
        results = {
            "experiment_name": self.config.experiment.name,
            "model": self.config.model.name_or_path,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "config": self.config.model_dump(),
            "variants": [r.to_dict() for r in self.results],
        }
        
        # Generate comparison summary
        results["comparison_summary"] = self._generate_summary()
        
        # Save results
        self._save_results(results)
        
        # Print summary
        self._print_summary()
        
        return results
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate comparison summary statistics."""
        summary = {
            "num_variants": len(self.results),
            "variants": [r.name for r in self.results],
        }
        
        if self.config.evaluation:
            # Extract key metrics for comparison
            for task in self.config.evaluation.tasks:
                task_results = {}
                for result in self.results:
                    if task in result.eval_results:
                        task_result = result.eval_results[task]
                        # Find accuracy metric
                        acc = None
                        for key in ["acc", "acc_norm", "exact_match", "f1"]:
                            if key in task_result:
                                acc = task_result[key]
                                break
                        if acc is not None:
                            task_results[result.name] = acc
                
                if task_results:
                    summary[f"task_{task}"] = task_results
                    # Compute relative change vs first variant (usually original)
                    baseline_name = list(task_results.keys())[0]
                    baseline_acc = task_results[baseline_name]
                    if baseline_acc > 0:
                        summary[f"task_{task}_relative"] = {
                            name: (acc - baseline_acc) / baseline_acc * 100
                            for name, acc in task_results.items()
                        }
        
        return summary
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save comparison results to disk."""
        results_path = self.output_dir / "comparison_results.json"
        
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_path}")
        
        # Also save config
        config_path = self.output_dir / "comparison_config.yaml"
        self.config.to_yaml(config_path)
    
    def _print_summary(self) -> None:
        """Print a formatted comparison summary."""
        logger.info("\n" + "=" * 80)
        logger.info("COMPARISON SUMMARY")
        logger.info("=" * 80)
        
        # Print table header
        header = f"{'Variant':<20} {'Method':<10} {'Noise':<12} {'Size(MB)':<12} {'Quant Time':<12}"
        if self.config.evaluation:
            header += " " + " ".join(f"{task:<12}" for task in self.config.evaluation.tasks)
        
        logger.info(header)
        logger.info("-" * len(header))
        
        # Print each variant
        for result in self.results:
            method = result.quantization_method or "original"
            noise = result.noise_function if result.noise_enabled else "none"
            quant_time = f"{result.quantization_time_seconds:.1f}s" if result.quantization_time_seconds > 0 else "N/A"
            
            row = f"{result.name:<20} {method:<10} {noise:<12} {result.model_size_mb:<12.1f} {quant_time:<12}"
            
            if self.config.evaluation:
                for task in self.config.evaluation.tasks:
                    if task in result.eval_results:
                        task_result = result.eval_results[task]
                        acc = None
                        # lm-eval returns keys with suffixes like "acc,none" or "acc_norm,none"
                        # Check both bare keys and keys with suffixes
                        metric_patterns = ["acc_norm", "acc", "exact_match", "f1"]
                        for metric in metric_patterns:
                            # Try exact match first
                            if metric in task_result:
                                acc = task_result[metric]
                                break
                            # Then try with ,none suffix (newer lm-eval format)
                            suffixed_key = f"{metric},none"
                            if suffixed_key in task_result:
                                acc = task_result[suffixed_key]
                                break
                        if acc is not None:
                            row += f" {acc:<12.4f}"
                        else:
                            row += f" {'N/A':<12}"
                    else:
                        row += f" {'N/A':<12}"
            
            logger.info(row)
        
        logger.info("=" * 80)


def load_comparison_config(path: Union[str, Path]) -> ComparisonExperimentConfig:
    """Load comparison experiment configuration from a YAML file.
    
    Args:
        path: Path to YAML configuration file
        
    Returns:
        Parsed ComparisonExperimentConfig
    """
    return ComparisonExperimentConfig.from_yaml(path)


def run_comparison(config: Union[str, Path, ComparisonExperimentConfig]) -> Dict[str, Any]:
    """Convenience function to run a comparison experiment.
    
    Args:
        config: Path to config file or ComparisonExperimentConfig object
        
    Returns:
        Comparison results
    """
    if isinstance(config, (str, Path)):
        config = load_comparison_config(config)
    
    runner = ComparisonRunner(config)
    return runner.run()


def main():
    """CLI entry point for running comparison experiments."""
    import argparse
    import os
    import sys
    
    # Suppress tokenizer parallelism warnings when forking for evaluation
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    
    from analog_ptq.utils.logging import setup_logging
    
    parser = argparse.ArgumentParser(
        description="Compare multiple quantization methods (Original, GPTQ, NA-GPTQ) with optional noise"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to comparison experiment configuration YAML file",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file",
    )
    parser.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=None,
        help="Specific variants to run (by name). If not specified, runs all enabled variants.",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level, log_file=args.log_file)
    
    # Load config
    try:
        config = load_comparison_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return 1
    
    # Filter variants if specified
    if args.variants:
        for variant in config.comparison.variants:
            if variant.name not in args.variants:
                variant.enabled = False
    
    # Run comparison
    try:
        results = run_comparison(config)
        logger.info("Comparison experiment completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"Comparison experiment failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

"""Experiment runner orchestrating the full quantization and evaluation pipeline."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

from analog_ptq.config.schema import ExperimentConfig, load_config
from analog_ptq.evaluation.harness import LMEvalHarness
from analog_ptq.evaluation.metrics import MetricsCollector
from analog_ptq.models.loader import load_model, load_tokenizer
from analog_ptq.models.wrapper import ModelWrapper
from analog_ptq.noise import WeightNoiseInjector, DynamicNoiseWrapper
from analog_ptq.pipeline.registry import registry
from analog_ptq.quantization.base import check_cached_model, load_cached_quantized_model
from analog_ptq.quantization.calibration import CalibrationDataLoader
from analog_ptq.utils.logging import get_logger, setup_logging


logger = get_logger(__name__)


class ExperimentRunner:
    """Orchestrates the full quantization and evaluation pipeline.
    
    Handles:
    - Loading models and tokenizers
    - Preparing calibration data
    - Running quantization
    - Applying noise injection (static and/or dynamic)
    - Evaluating on benchmarks
    - Saving results and models
    
    Example:
        >>> config = load_config("configs/my_experiment.yaml")
        >>> runner = ExperimentRunner(config)
        >>> results = runner.run()
    """
    
    def __init__(self, config: ExperimentConfig):
        """Initialize the experiment runner.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.metrics = MetricsCollector()
        
        # Create output directory
        self.output_dir = Path(config.experiment.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set random seed
        self._set_seed(config.experiment.seed)
        
        # Placeholders
        self.model = None
        self.tokenizer = None
        self.wrapper = None
        self.noise_injector = None
        self.dynamic_noise_wrapper = None
    
    def _set_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility.
        
        Args:
            seed: Random seed
        """
        import random
        import numpy as np
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def load_model(self) -> ModelWrapper:
        """Load the model and tokenizer.
        
        Returns:
            ModelWrapper containing the loaded model
        """
        logger.info("=" * 50)
        logger.info("Loading model...")
        logger.info("=" * 50)
        
        model_config = self.config.model
        
        self.model = load_model(
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
        
        self.wrapper = ModelWrapper(self.model)
        
        # Measure baseline metrics
        self.metrics.measure_model_size(self.model)
        
        return self.wrapper
    
    def prepare_calibration_data(self) -> list:
        """Prepare calibration data for quantization.
        
        Returns:
            List of calibration tensors
        """
        if self.config.quantization is None:
            return []
        
        logger.info("=" * 50)
        logger.info("Preparing calibration data...")
        logger.info("=" * 50)
        
        calib_config = self.config.quantization.calibration
        
        loader = CalibrationDataLoader(
            tokenizer=self.tokenizer,
            seq_length=calib_config.seq_length,
        )
        
        samples = loader.load_dataset_samples(
            dataset_name=calib_config.dataset,
            num_samples=calib_config.num_samples,
            seed=calib_config.seed,
        )
        
        return samples
    
    def quantize(self, calibration_data: list) -> ModelWrapper:
        """Apply quantization to the model.
        
        Args:
            calibration_data: Calibration samples
            
        Returns:
            Quantized model wrapper
        """
        if self.config.quantization is None:
            logger.info("No quantization configured, skipping...")
            return self.wrapper
        
        logger.info("=" * 50)
        logger.info("Starting quantization...")
        logger.info("=" * 50)
        
        quant_config = self.config.quantization
        quant_output = self.output_dir / "quantized_model"
        force_requantize = self.config.experiment.force_requantize
        
        # Check for cached model first
        if check_cached_model(quant_output) and not force_requantize:
            logger.info(f"Found cached quantized model at {quant_output}")
            logger.info("Loading cached model (use force_requantize: true to re-quantize)")
            
            try:
                model_config = self.config.model
                device_map = model_config.device_map
                
                model, quant_config_dict = load_cached_quantized_model(
                    quant_output,
                    device_map=device_map,
                    dtype=model_config.dtype,
                )
                self.wrapper = ModelWrapper(model)
                
                # Measure post-load metrics
                self.metrics.measure_model_size(self.wrapper)
                self.metrics.count_quantized_parameters(self.wrapper)
                
                logger.info("Cached model loaded successfully")
                return self.wrapper
                
            except Exception as e:
                logger.warning(f"Failed to load cached model: {e}")
                logger.info("Falling back to re-quantization...")
        
        # Get quantizer from registry
        quantizer_cls = registry.get_quantizer(quant_config.method)
        
        # Create quantizer instance
        quantizer = quantizer_cls(
            bits=quant_config.bits,
            group_size=quant_config.group_size,
            symmetric=quant_config.symmetric,
            damp_percent=quant_config.damp_percent,
            block_size=quant_config.block_size,
            actorder=quant_config.actorder,
            **quant_config.extra,
        )
        
        # Prepare and quantize
        self.wrapper = quantizer.prepare(self.wrapper)
        self.wrapper = quantizer.quantize(self.wrapper, calibration_data)
        
        # Measure post-quantization metrics
        self.metrics.measure_model_size(self.wrapper)
        self.metrics.count_quantized_parameters(self.wrapper)
        
        # Verify weight preservation (should be same before/after quantization)
        self.metrics.verify_weight_preservation(
            self.metrics._original_linear_weights, 
            self.wrapper
        )
        
        # Log detailed breakdown in debug mode
        import os
        if os.environ.get("NAGPTQ_DEBUG", "0") == "1":
            self.metrics.log_parameter_breakdown(self.wrapper)
        
        # Save quantized model
        quantizer.save(
            self.wrapper, 
            quant_output,
            original_model=self.config.model.name_or_path,
        )
        
        return self.wrapper
    
    def apply_noise(self) -> ModelWrapper:
        """Apply noise injection to the model.
        
        Supports three modes:
        - "static": Permanent noise added to weights
        - "dynamic": Noise added during each forward pass
        - "both": Both static and dynamic noise
        
        Returns:
            Model wrapper with noise applied
        """
        if self.config.noise is None or not self.config.noise.enabled:
            logger.info("No noise injection configured, skipping...")
            return self.wrapper
        
        logger.info("=" * 50)
        logger.info("Applying noise injection...")
        logger.info("=" * 50)
        
        noise_config = self.config.noise
        mode = noise_config.mode
        
        logger.info(f"  Mode: {mode}")
        logger.info(f"  Function: {noise_config.function}")
        logger.info(f"  Parameters: {noise_config.function_params}")
        
        # Apply static noise (permanent weight perturbation)
        if mode in ("static", "both"):
            logger.info("Applying static noise...")
            self.noise_injector = WeightNoiseInjector(
                noise_function=noise_config.function,
                function_params=noise_config.function_params,
                seed=noise_config.seed,
                target_layers=noise_config.target_layers,
            )
            self.wrapper = self.noise_injector.apply(self.wrapper)
            
            # Log statistics
            logger.info(self.noise_injector.summary())
            
            # Add noise stats to metrics
            self.metrics.add_custom_metric(
                "noise_injection",
                {
                    "mode": mode,
                    "function": noise_config.function,
                    "function_params": noise_config.function_params,
                    "stats": self.noise_injector.get_stats(),
                }
            )
        
        # Apply dynamic noise (inference-time noise)
        if mode in ("dynamic", "both"):
            logger.info("Applying dynamic noise wrapper...")
            self.dynamic_noise_wrapper = DynamicNoiseWrapper(
                noise_function=noise_config.function,
                function_params=noise_config.function_params,
                target_layers=noise_config.target_layers,
            )
            self.wrapper = self.dynamic_noise_wrapper.apply(self.wrapper)
            
            logger.info(f"  Wrapped {len(self.dynamic_noise_wrapper.get_wrapped_layers())} layers")
        
        return self.wrapper
    
    def evaluate(self) -> Dict[str, Any]:
        """Run evaluation on the model.
        
        Returns:
            Evaluation results dictionary
        """
        if self.config.evaluation is None:
            logger.info("No evaluation configured, skipping...")
            return {}
        
        logger.info("=" * 50)
        logger.info("Running evaluation...")
        logger.info("=" * 50)
        
        eval_config = self.config.evaluation
        
        harness = LMEvalHarness(
            model=self.wrapper,
            tokenizer=self.tokenizer,
            device=eval_config.device,
        )
        
        results = harness.evaluate(
            tasks=eval_config.tasks,
            num_fewshot=eval_config.num_fewshot,
            batch_size=eval_config.batch_size,
            limit=eval_config.limit,
        )
        
        self.metrics.add_eval_results(results)
        
        return results
    
    def run(self) -> Dict[str, Any]:
        """Run the full experiment pipeline.
        
        Returns:
            Dictionary containing all results
        """
        logger.info("=" * 50)
        logger.info(f"Starting experiment: {self.config.experiment.name}")
        logger.info("=" * 50)
        
        start_time = datetime.now()
        
        # Load model
        self.load_model()
        
        # Quantization (if configured)
        if self.config.quantization:
            calibration_data = self.prepare_calibration_data()
            self.quantize(calibration_data)
        
        # Noise injection (if configured)
        if self.config.noise:
            self.apply_noise()
        
        # Evaluation (if configured)
        eval_results = {}
        if self.config.evaluation:
            eval_results = self.evaluate()
        
        # Collect final metrics
        self.metrics.measure_memory_usage()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Prepare results
        results = {
            "experiment_name": self.config.experiment.name,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "config": self.config.model_dump(),
            "metrics": self.metrics.get_metrics().to_dict(),
            "evaluation": eval_results.get("results", {}),
        }
        
        # Save results
        self._save_results(results)
        
        # Print summary
        logger.info("\n" + self.metrics.summary())
        
        return results
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save experiment results to disk.
        
        Args:
            results: Results dictionary
        """
        results_path = self.output_dir / "results.json"
        
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_path}")
        
        # Also save the config
        config_path = self.output_dir / "config.yaml"
        self.config.to_yaml(config_path)


def run_experiment(config: Union[str, Path, ExperimentConfig]) -> Dict[str, Any]:
    """Convenience function to run an experiment.
    
    Args:
        config: Path to config file or ExperimentConfig object
        
    Returns:
        Experiment results
        
    Example:
        >>> results = run_experiment("configs/my_experiment.yaml")
    """
    if isinstance(config, (str, Path)):
        config = load_config(config)
    
    runner = ExperimentRunner(config)
    return runner.run()


def main():
    """CLI entry point for running experiments."""
    import os
    
    # Suppress tokenizer parallelism warnings when forking for evaluation
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    
    parser = argparse.ArgumentParser(
        description="Run quantization and evaluation experiments"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to experiment configuration YAML file",
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
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level, log_file=args.log_file)
    
    # Run experiment
    try:
        results = run_experiment(args.config)
        logger.info("Experiment completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

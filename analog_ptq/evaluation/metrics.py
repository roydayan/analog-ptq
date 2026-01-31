"""Metrics collection for model evaluation."""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import PreTrainedModel

from analog_ptq.models.wrapper import ModelWrapper
from analog_ptq.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics.
    
    Model Size Metrics:
        model_size_mb: Actual memory footprint of the model in MB
        original_params: Number of parameters in the original (pre-quantization) model
        quantized_weights: Number of weights in linear layers that were quantized
        num_quantized_layers: Number of linear layers that were replaced with QuantizedLinear
    
    Memory Metrics:
        peak_memory_mb: Peak GPU memory usage during experiment
    
    Timing Metrics:
        load_time_s: Time to load the model
        inference_time_s: Average inference time per forward pass
        tokens_per_second: Throughput in tokens per second
    """
    
    # Memory metrics
    model_size_mb: float = 0.0
    peak_memory_mb: float = 0.0
    
    # Timing metrics
    load_time_s: float = 0.0
    inference_time_s: float = 0.0
    tokens_per_second: float = 0.0
    
    # Model metrics - BEFORE quantization
    original_params: int = 0  # Total params in original model
    
    # Model metrics - AFTER quantization
    quantized_weights: int = 0      # Number of weights in quantized linear layers
    num_quantized_layers: int = 0   # Number of QuantizedLinear layers
    
    # Legacy field names for backward compatibility
    @property
    def num_parameters(self) -> int:
        return self.original_params
    
    @property
    def num_quantized_params(self) -> int:
        return self.quantized_weights
    
    # Accuracy metrics (from evaluation)
    eval_results: Dict[str, Any] = field(default_factory=dict)
    
    # Custom metrics (e.g., noise injection stats)
    custom: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "model_size_mb": self.model_size_mb,
            "peak_memory_mb": self.peak_memory_mb,
            "load_time_s": self.load_time_s,
            "inference_time_s": self.inference_time_s,
            "tokens_per_second": self.tokens_per_second,
            "original_params": self.original_params,
            "quantized_weights": self.quantized_weights,
            "num_quantized_layers": self.num_quantized_layers,
            "eval_results": self.eval_results,
            "custom": self.custom,
        }


class MetricsCollector:
    """Collects various metrics during model evaluation.
    
    Tracks:
    - Model size and memory usage
    - Inference latency
    - Throughput (tokens/second)
    - Quantization statistics
    """
    
    def __init__(self):
        """Initialize the metrics collector."""
        self.metrics = PerformanceMetrics()
        self._start_time: Optional[float] = None
        self._inference_times: List[float] = []
        self._total_tokens: int = 0
        self._original_params_measured: bool = False
        self._original_linear_weights: int = 0
    
    def count_linear_weights(
        self,
        model: Union[PreTrainedModel, ModelWrapper],
    ) -> int:
        """Count total weights in all linear layers.
        
        This counts in_features × out_features for each nn.Linear layer.
        Used to verify weight preservation after quantization.
        
        Args:
            model: Model to analyze
            
        Returns:
            Total number of weights in linear layers
        """
        import torch.nn as nn
        
        if isinstance(model, ModelWrapper):
            model = model.model
        
        total_weights = 0
        for module in model.modules():
            if isinstance(module, nn.Linear):
                total_weights += module.in_features * module.out_features
        
        return total_weights
    
    def measure_model_size(
        self,
        model: Union[PreTrainedModel, ModelWrapper],
    ) -> float:
        """Measure model size in megabytes.
        
        This measures the actual memory footprint of the model, including:
        - All parameters (weights, biases)
        - All buffers (quantized weights, scales, zeros, etc.)
        
        For quantized models, this reflects the compressed size.
        
        Args:
            model: Model to measure
            
        Returns:
            Model size in MB
        """
        if isinstance(model, ModelWrapper):
            model = model.model
        
        # Count total size in bytes
        total_size = 0
        total_params = 0
        
        for param in model.parameters():
            total_params += param.numel()
            total_size += param.numel() * param.element_size()
        
        # Also count buffers (includes qweight, scales, zeros in QuantizedLinear)
        for buffer in model.buffers():
            total_size += buffer.numel() * buffer.element_size()
        
        size_mb = total_size / (1024 * 1024)
        
        self.metrics.model_size_mb = size_mb
        
        # Only store original_params on first call (before quantization)
        if not self._original_params_measured:
            self.metrics.original_params = total_params
            self._original_params_measured = True
            # Also count original linear weights for verification
            self._original_linear_weights = self.count_linear_weights(model)
            logger.info(f"Original model: {size_mb:.2f} MB, {total_params:,} parameters")
            logger.info(f"  Linear layer weights: {self._original_linear_weights:,}")
        else:
            logger.info(f"Quantized model size: {size_mb:.2f} MB")
        
        return size_mb
    
    def measure_memory_usage(self) -> float:
        """Measure current GPU memory usage.
        
        Returns:
            Peak memory usage in MB (or 0 if no GPU)
        """
        if not torch.cuda.is_available():
            return 0.0
        
        peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
        self.metrics.peak_memory_mb = peak_memory
        
        return peak_memory
    
    def start_timer(self) -> None:
        """Start the inference timer."""
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self._start_time = time.perf_counter()
    
    def stop_timer(self, num_tokens: int = 0) -> float:
        """Stop the timer and record the elapsed time.
        
        Args:
            num_tokens: Number of tokens processed
            
        Returns:
            Elapsed time in seconds
        """
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        if self._start_time is None:
            return 0.0
        
        elapsed = time.perf_counter() - self._start_time
        self._inference_times.append(elapsed)
        self._total_tokens += num_tokens
        
        self._start_time = None
        
        return elapsed
    
    def measure_inference_latency(
        self,
        model: Union[PreTrainedModel, ModelWrapper],
        input_ids: torch.Tensor,
        num_runs: int = 10,
        warmup_runs: int = 3,
    ) -> Dict[str, float]:
        """Measure inference latency with multiple runs.
        
        Args:
            model: Model to benchmark
            input_ids: Input tensor
            num_runs: Number of timed runs
            warmup_runs: Number of warmup runs
            
        Returns:
            Dictionary with latency statistics
        """
        if isinstance(model, ModelWrapper):
            model = model.model
        
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(input_ids)
        
        # Timed runs
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                self.start_timer()
                _ = model(input_ids)
                elapsed = self.stop_timer(input_ids.numel())
                times.append(elapsed)
        
        # Compute statistics
        mean_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        self.metrics.inference_time_s = mean_time
        
        tokens_per_second = input_ids.numel() / mean_time if mean_time > 0 else 0
        self.metrics.tokens_per_second = tokens_per_second
        
        return {
            "mean_s": mean_time,
            "min_s": min_time,
            "max_s": max_time,
            "tokens_per_second": tokens_per_second,
        }
    
    def count_quantized_parameters(
        self,
        model: Union[PreTrainedModel, ModelWrapper],
    ) -> int:
        """Count the number of weights in quantized linear layers.
        
        This counts the ORIGINAL number of weight elements (in_features × out_features)
        for each QuantizedLinear layer. This represents how many floating-point weights
        were converted to quantized format.
        
        For example, a linear layer with shape [4096, 4096] contributes 16,777,216
        to this count, regardless of how those weights are stored after quantization.
        
        Args:
            model: Model to analyze
            
        Returns:
            Number of weights in quantized layers
        """
        from analog_ptq.quantization.utils import QuantizedLinear
        
        if isinstance(model, ModelWrapper):
            model = model.model
        
        quantized_weights = 0
        num_layers = 0
        
        for module in model.modules():
            if isinstance(module, QuantizedLinear):
                quantized_weights += module.in_features * module.out_features
                num_layers += 1
        
        self.metrics.quantized_weights = quantized_weights
        self.metrics.num_quantized_layers = num_layers
        
        logger.info(f"Quantized {num_layers} linear layers ({quantized_weights:,} weights)")
        
        return quantized_weights
    
    def detailed_parameter_breakdown(
        self,
        model: Union[PreTrainedModel, ModelWrapper],
    ) -> Dict[str, Any]:
        """Get detailed breakdown of model components.
        
        This helps understand exactly what's in the model after quantization
        and verify that the weight counts match.
        
        Args:
            model: Model to analyze
            
        Returns:
            Dictionary with detailed breakdown
        """
        from analog_ptq.quantization.utils import QuantizedLinear
        import torch.nn as nn
        
        if isinstance(model, ModelWrapper):
            model = model.model
        
        breakdown = {
            "quantized_linear": {
                "count": 0,
                "weights": 0,           # in_features × out_features (the actual weight matrix size)
                "qweight_stored": 0,    # How qweight is actually stored (may include padding)
                "scales_stored": 0,
                "zeros_stored": 0,
                "bias": 0,
            },
            "regular_linear": {
                "count": 0,
                "weights": 0,           # in_features × out_features
                "bias": 0,
            },
            "embedding": {"count": 0, "weights": 0},
            "layernorm": {"count": 0, "params": 0},
        }
        
        for name, module in model.named_modules():
            if isinstance(module, QuantizedLinear):
                breakdown["quantized_linear"]["count"] += 1
                # The actual weight matrix size (what we're quantizing)
                breakdown["quantized_linear"]["weights"] += module.in_features * module.out_features
                # How it's stored (may be padded for group alignment)
                breakdown["quantized_linear"]["qweight_stored"] += module.qweight.numel()
                breakdown["quantized_linear"]["scales_stored"] += module.scales.numel()
                breakdown["quantized_linear"]["zeros_stored"] += module.zeros.numel()
                if module.bias is not None:
                    breakdown["quantized_linear"]["bias"] += module.bias.numel()
                
            elif isinstance(module, nn.Linear):
                breakdown["regular_linear"]["count"] += 1
                breakdown["regular_linear"]["weights"] += module.in_features * module.out_features
                if module.bias is not None:
                    breakdown["regular_linear"]["bias"] += module.bias.numel()
                    
            elif isinstance(module, nn.Embedding):
                breakdown["embedding"]["count"] += 1
                breakdown["embedding"]["weights"] += module.weight.numel()
                
            elif isinstance(module, (nn.LayerNorm,)) or "LayerNorm" in type(module).__name__ or "RMSNorm" in type(module).__name__:
                breakdown["layernorm"]["count"] += 1
                for param in module.parameters():
                    breakdown["layernorm"]["params"] += param.numel()
        
        # Compute totals
        # Total weights in linear layers (should match before/after quantization)
        total_linear_weights = (
            breakdown["quantized_linear"]["weights"] + 
            breakdown["regular_linear"]["weights"]
        )
        
        breakdown["totals"] = {
            "total_linear_weights": total_linear_weights,
            "quantized_linear_weights": breakdown["quantized_linear"]["weights"],
            "unquantized_linear_weights": breakdown["regular_linear"]["weights"],
            "embedding_weights": breakdown["embedding"]["weights"],
            "layernorm_params": breakdown["layernorm"]["params"],
        }
        
        return breakdown
    
    def log_parameter_breakdown(
        self,
        model: Union[PreTrainedModel, ModelWrapper],
    ) -> None:
        """Log a detailed parameter breakdown for debugging.
        
        This verifies that the weight counts are consistent and shows
        exactly what's in the quantized model.
        
        Args:
            model: Model to analyze
        """
        breakdown = self.detailed_parameter_breakdown(model)
        
        logger.info("=" * 60)
        logger.info("Model Parameter Breakdown (Post-Quantization)")
        logger.info("=" * 60)
        
        ql = breakdown["quantized_linear"]
        if ql["count"] > 0:
            logger.info(f"Quantized Linear Layers: {ql['count']}")
            logger.info(f"  Weight matrix size (in×out): {ql['weights']:,}")
            logger.info(f"  Stored as:")
            logger.info(f"    - qweight (int8):  {ql['qweight_stored']:,} values")
            logger.info(f"    - scales (fp16):   {ql['scales_stored']:,} values")
            logger.info(f"    - zeros (fp16):    {ql['zeros_stored']:,} values")
            if ql["bias"] > 0:
                logger.info(f"    - bias:            {ql['bias']:,} values")
        
        rl = breakdown["regular_linear"]
        if rl["count"] > 0:
            logger.info(f"Unquantized Linear Layers: {rl['count']} (e.g., lm_head)")
            logger.info(f"  Weight matrix size: {rl['weights']:,}")
            if rl["bias"] > 0:
                logger.info(f"  Bias: {rl['bias']:,}")
        
        emb = breakdown["embedding"]
        if emb["count"] > 0:
            logger.info(f"Embedding Layers: {emb['count']}")
            logger.info(f"  Weights: {emb['weights']:,}")
        
        ln = breakdown["layernorm"]
        if ln["count"] > 0:
            logger.info(f"LayerNorm/RMSNorm: {ln['count']}")
            logger.info(f"  Parameters: {ln['params']:,}")
        
        totals = breakdown["totals"]
        logger.info("-" * 60)
        logger.info("Weight Count Summary:")
        logger.info(f"  Total linear layer weights:    {totals['total_linear_weights']:,}")
        logger.info(f"    - Quantized:                 {totals['quantized_linear_weights']:,}")
        logger.info(f"    - Unquantized:               {totals['unquantized_linear_weights']:,}")
        logger.info(f"  Embedding weights:             {totals['embedding_weights']:,}")
        logger.info(f"  LayerNorm parameters:          {totals['layernorm_params']:,}")
        logger.info("=" * 60)
    
    def verify_weight_preservation(
        self,
        original_linear_weights: int,
        model: Union[PreTrainedModel, ModelWrapper],
    ) -> bool:
        """Verify that quantization preserved the same number of weights.
        
        This checks that the total weights in linear layers is the same
        before and after quantization (just stored differently).
        
        Args:
            original_linear_weights: Weight count from original model
            model: Quantized model to check
            
        Returns:
            True if weights match, False otherwise
        """
        breakdown = self.detailed_parameter_breakdown(model)
        quantized_total = breakdown["totals"]["total_linear_weights"]
        
        if original_linear_weights == quantized_total:
            logger.info(f"✓ Weight count verified: {quantized_total:,} linear weights preserved")
            return True
        else:
            logger.warning(f"✗ Weight count mismatch!")
            logger.warning(f"  Original: {original_linear_weights:,}")
            logger.warning(f"  After quantization: {quantized_total:,}")
            logger.warning(f"  Difference: {quantized_total - original_linear_weights:,}")
            return False
    
    def add_eval_results(self, results: Dict[str, Any]) -> None:
        """Add evaluation results to metrics.
        
        Args:
            results: Evaluation results from LMEvalHarness
        """
        if "results" in results:
            self.metrics.eval_results = results["results"]
        else:
            self.metrics.eval_results = results
    
    def add_custom_metric(self, name: str, value: Any) -> None:
        """Add a custom metric.
        
        Args:
            name: Metric name
            value: Metric value (can be any serializable object)
        """
        self.metrics.custom[name] = value
    
    def get_metrics(self) -> PerformanceMetrics:
        """Get the collected metrics.
        
        Returns:
            PerformanceMetrics dataclass
        """
        return self.metrics
    
    def reset(self) -> None:
        """Reset all collected metrics."""
        self.metrics = PerformanceMetrics()
        self._start_time = None
        self._inference_times = []
        self._total_tokens = 0
        self._original_params_measured = False
    
    def summary(self) -> str:
        """Generate a human-readable summary of metrics.
        
        Returns:
            Summary string
        """
        m = self.metrics
        
        lines = [
            "=" * 50,
            "Performance Metrics Summary",
            "=" * 50,
        ]
        
        # Model size info
        lines.append(f"Model Size: {m.model_size_mb:.2f} MB")
        
        if m.original_params > 0:
            lines.append(f"Original Model Parameters: {m.original_params:,}")
        
        # Quantization info
        if m.quantized_weights > 0:
            lines.append(f"Quantized Linear Layers: {m.num_quantized_layers}")
            lines.append(f"Weights in Quantized Layers: {m.quantized_weights:,}")
            
            # Compute what percentage of model was quantized
            if m.original_params > 0:
                pct = 100 * m.quantized_weights / m.original_params
                lines.append(f"Quantization Coverage: {pct:.1f}% of weights")
        
        # Memory and performance
        lines.append(f"Peak GPU Memory: {m.peak_memory_mb:.2f} MB")
        
        if m.inference_time_s > 0:
            lines.append(f"Inference Time: {m.inference_time_s * 1000:.2f} ms")
            lines.append(f"Throughput: {m.tokens_per_second:.2f} tokens/s")
        
        # Evaluation results
        if m.eval_results:
            lines.append("-" * 50)
            lines.append("Evaluation Results:")
            for task, result in m.eval_results.items():
                if isinstance(result, dict):
                    acc = result.get("acc", result.get("acc_norm", "N/A"))
                    lines.append(f"  {task}: {acc}")
        
        lines.append("=" * 50)
        
        return "\n".join(lines)

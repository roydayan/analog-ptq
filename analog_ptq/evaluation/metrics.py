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
    """Container for performance metrics."""
    
    # Memory metrics
    model_size_mb: float = 0.0
    peak_memory_mb: float = 0.0
    
    # Timing metrics
    load_time_s: float = 0.0
    inference_time_s: float = 0.0
    tokens_per_second: float = 0.0
    
    # Model metrics
    num_parameters: int = 0
    num_quantized_params: int = 0
    
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
            "num_parameters": self.num_parameters,
            "num_quantized_params": self.num_quantized_params,
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
    
    def measure_model_size(
        self,
        model: Union[PreTrainedModel, ModelWrapper],
    ) -> float:
        """Measure model size in megabytes.
        
        Args:
            model: Model to measure
            
        Returns:
            Model size in MB
        """
        if isinstance(model, ModelWrapper):
            model = model.model
        
        # Count parameters
        total_params = 0
        total_size = 0
        
        for param in model.parameters():
            total_params += param.numel()
            total_size += param.numel() * param.element_size()
        
        # Also count buffers
        for buffer in model.buffers():
            total_size += buffer.numel() * buffer.element_size()
        
        size_mb = total_size / (1024 * 1024)
        
        self.metrics.model_size_mb = size_mb
        self.metrics.num_parameters = total_params
        
        logger.info(f"Model size: {size_mb:.2f} MB, Parameters: {total_params:,}")
        
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
        """Count the number of quantized parameters.
        
        Args:
            model: Model to analyze
            
        Returns:
            Number of quantized parameters
        """
        from analog_ptq.quantization.utils import QuantizedLinear
        
        if isinstance(model, ModelWrapper):
            model = model.model
        
        quantized_params = 0
        
        for module in model.modules():
            if isinstance(module, QuantizedLinear):
                quantized_params += module.in_features * module.out_features
        
        self.metrics.num_quantized_params = quantized_params
        
        return quantized_params
    
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
    
    def summary(self) -> str:
        """Generate a human-readable summary of metrics.
        
        Returns:
            Summary string
        """
        lines = [
            "=" * 50,
            "Performance Metrics Summary",
            "=" * 50,
            f"Model Size: {self.metrics.model_size_mb:.2f} MB",
            f"Parameters: {self.metrics.num_parameters:,}",
            f"Quantized Parameters: {self.metrics.num_quantized_params:,}",
            f"Peak Memory: {self.metrics.peak_memory_mb:.2f} MB",
            f"Inference Time: {self.metrics.inference_time_s * 1000:.2f} ms",
            f"Throughput: {self.metrics.tokens_per_second:.2f} tokens/s",
        ]
        
        if self.metrics.eval_results:
            lines.append("-" * 50)
            lines.append("Evaluation Results:")
            for task, result in self.metrics.eval_results.items():
                if isinstance(result, dict):
                    acc = result.get("acc", result.get("acc_norm", "N/A"))
                    lines.append(f"  {task}: {acc}")
        
        lines.append("=" * 50)
        
        return "\n".join(lines)

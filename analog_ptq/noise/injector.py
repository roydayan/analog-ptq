"""Weight noise injector for applying static Gaussian noise to quantized model weights.

This module provides the WeightNoiseInjector class that applies permanent Gaussian noise
to model weights, where the noise standard deviation is computed via a configurable function.
"""

from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from analog_ptq.models.wrapper import ModelWrapper
from analog_ptq.noise.functions import get_noise_function, NoiseFunctionType
from analog_ptq.quantization.utils import QuantizedLinear
from analog_ptq.utils.logging import get_logger


logger = get_logger(__name__)


class WeightNoiseInjector:
    """Applies static Gaussian noise to model weights.
    
    The noise standard deviation for each weight is computed using a configurable
    function sigma = f(weight), where f can be any registered noise function.
    
    This injector permanently modifies the weights by adding sampled noise.
    For dynamic noise at inference time, use DynamicNoiseWrapper instead.
    
    Example:
        >>> injector = WeightNoiseInjector(
        ...     noise_function="proportional",
        ...     function_params={"scale": 0.05},
        ...     seed=42,
        ... )
        >>> model = injector.apply(model)
    """
    
    def __init__(
        self,
        noise_function: Union[str, NoiseFunctionType] = "proportional",
        function_params: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        target_layers: Optional[List[str]] = None,
        quantized_only: bool = True,
    ):
        """Initialize the noise injector.
        
        Args:
            noise_function: Name of registered noise function or a callable.
                           The function should take (weight, **params) and return sigma tensor.
            function_params: Parameters to pass to the noise function
            seed: Random seed for reproducibility
            target_layers: Optional list of layer name patterns to target.
                          If None, applies to all quantized linear layers.
            quantized_only: If True (default), only apply noise to QuantizedLinear layers.
                          This is the correct behavior for simulating analog hardware noise,
                          as only quantized layers represent the analog weights.
                          Set to False to also apply noise to regular nn.Linear layers.
        """
        if isinstance(noise_function, str):
            self.noise_fn = get_noise_function(noise_function)
            self.noise_fn_name = noise_function
        else:
            self.noise_fn = noise_function
            self.noise_fn_name = getattr(noise_function, "__name__", "custom")
        
        self.function_params = function_params or {}
        self.seed = seed
        self.target_layers = target_layers
        self.quantized_only = quantized_only
        
        # Statistics tracking
        self._stats: Dict[str, Dict[str, float]] = {}
        
        logger.info(f"Initialized WeightNoiseInjector")
        logger.info(f"  function={self.noise_fn_name}, params={self.function_params}")
        logger.info(f"  quantized_only={self.quantized_only}")
    
    def apply(
        self,
        model: Union[PreTrainedModel, ModelWrapper],
    ) -> Union[PreTrainedModel, ModelWrapper]:
        """Apply Gaussian noise to model weights.
        
        Iterates through all quantized linear layers (or targeted layers) and adds
        Gaussian noise to the weights based on the configured sigma function.
        
        Args:
            model: The model to add noise to
            
        Returns:
            The same model with noise added to weights
        """
        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
        
        # Get the underlying model if wrapped
        if isinstance(model, ModelWrapper):
            hf_model = model.model
        else:
            hf_model = model
        
        total_layers = 0
        total_params = 0
        
        for name, module in hf_model.named_modules():
            if not self._should_process_layer(name, module):
                continue
            
            if isinstance(module, QuantizedLinear):
                params_modified = self._apply_to_quantized_linear(name, module)
            elif isinstance(module, nn.Linear):
                params_modified = self._apply_to_linear(name, module)
            else:
                continue
            
            total_layers += 1
            total_params += params_modified
        
        logger.info(f"Applied noise to {total_layers} layers ({total_params:,} parameters)")
        
        return model
    
    def _should_process_layer(self, name: str, module: nn.Module) -> bool:
        """Check if a layer should have noise applied.
        
        Args:
            name: Layer name
            module: Layer module
            
        Returns:
            True if noise should be applied to this layer
        """
        # Check layer type based on quantized_only flag
        if self.quantized_only:
            # Only process QuantizedLinear layers (analog hardware simulation)
            if not isinstance(module, QuantizedLinear):
                return False
        else:
            # Process both Linear and QuantizedLinear layers
            if not isinstance(module, (nn.Linear, QuantizedLinear)):
                return False
        
        # If no target patterns specified, process all eligible layers
        if self.target_layers is None:
            return True
        
        # Check if name matches any target pattern
        return any(pattern in name for pattern in self.target_layers)
    
    def _apply_to_quantized_linear(
        self,
        name: str,
        layer: QuantizedLinear,
    ) -> int:
        """Apply noise to a QuantizedLinear layer.
        
        For quantized layers, we dequantize the weights, add noise, and update
        the quantized representation.
        
        Args:
            name: Layer name
            layer: QuantizedLinear layer
            
        Returns:
            Number of parameters modified
        """
        # Dequantize to get float weights
        weight = layer._dequantize_weight()
        weight = weight[:, :layer.in_features]  # Trim padding
        
        # Compute sigma for each weight
        sigma = self.noise_fn(weight, **self.function_params)
        
        # Sample noise
        noise = torch.randn_like(weight) * sigma
        
        # Add noise to weights
        noisy_weight = weight + noise
        
        # Re-quantize and update the layer's buffers
        self._update_quantized_layer(layer, noisy_weight)
        
        # Track statistics
        self._stats[name] = {
            "mean_sigma": sigma.mean().item(),
            "max_sigma": sigma.max().item(),
            "mean_noise": noise.abs().mean().item(),
            "weight_std_before": weight.std().item(),
            "weight_std_after": noisy_weight.std().item(),
        }
        
        logger.debug(f"  {name}: applied noise (mean_sigma={sigma.mean().item():.6f})")
        
        return weight.numel()
    
    def _apply_to_linear(
        self,
        name: str,
        layer: nn.Linear,
    ) -> int:
        """Apply noise to a regular Linear layer.
        
        Args:
            name: Layer name
            layer: Linear layer
            
        Returns:
            Number of parameters modified
        """
        weight = layer.weight.data
        
        # Compute sigma for each weight
        sigma = self.noise_fn(weight, **self.function_params)
        
        # Sample noise
        noise = torch.randn_like(weight) * sigma
        
        # Add noise to weights
        layer.weight.data = weight + noise
        
        # Track statistics
        self._stats[name] = {
            "mean_sigma": sigma.mean().item(),
            "max_sigma": sigma.max().item(),
            "mean_noise": noise.abs().mean().item(),
        }
        
        logger.debug(f"  {name}: applied noise (mean_sigma={sigma.mean().item():.6f})")
        
        return weight.numel()
    
    def _update_quantized_layer(
        self,
        layer: QuantizedLinear,
        new_weight: torch.Tensor,
    ) -> None:
        """Update a QuantizedLinear layer with new weights.
        
        Re-quantizes the weights and updates the layer's buffers.
        
        Args:
            layer: QuantizedLinear layer to update
            new_weight: New float weights to quantize
        """
        from analog_ptq.quantization.utils import quantize_tensor
        
        # Pad weight if needed
        padded_size = layer.num_groups * layer.group_size
        if new_weight.shape[1] < padded_size:
            new_weight = torch.nn.functional.pad(
                new_weight, (0, padded_size - new_weight.shape[1])
            )
        
        # Re-quantize
        qweight, scales, zeros = quantize_tensor(
            new_weight,
            bits=layer.bits,
            group_size=layer.group_size,
            symmetric=False,
        )
        
        # Update buffers
        layer.qweight.copy_(qweight.view(layer.qweight.shape).to(torch.int8))
        layer.scales.copy_(scales.view(layer.scales.shape).to(torch.float16))
        if zeros is not None:
            layer.zeros.copy_(zeros.view(layer.zeros.shape).to(torch.float16))
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics about the noise injection.
        
        Returns:
            Dictionary mapping layer names to their noise statistics
        """
        return self._stats.copy()
    
    def summary(self) -> str:
        """Get a summary of noise injection statistics.
        
        Returns:
            Formatted summary string
        """
        if not self._stats:
            return "No noise applied yet."
        
        lines = ["Noise Injection Summary:"]
        lines.append(f"  Function: {self.noise_fn_name}")
        lines.append(f"  Parameters: {self.function_params}")
        lines.append(f"  Layers processed: {len(self._stats)}")
        
        # Aggregate statistics
        all_mean_sigmas = [s["mean_sigma"] for s in self._stats.values()]
        all_max_sigmas = [s["max_sigma"] for s in self._stats.values()]
        
        lines.append(f"  Avg mean sigma: {sum(all_mean_sigmas)/len(all_mean_sigmas):.6f}")
        lines.append(f"  Max sigma: {max(all_max_sigmas):.6f}")
        
        return "\n".join(lines)

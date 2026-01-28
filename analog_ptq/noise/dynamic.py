"""Dynamic noise wrapper for applying Gaussian noise at inference time.

This module provides the DynamicNoiseWrapper class that wraps QuantizedLinear layers
to apply fresh Gaussian noise during each forward pass, simulating analog compute noise.
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


class NoisyQuantizedLinear(nn.Module):
    """A wrapper around QuantizedLinear that adds noise at inference time.
    
    This module wraps a QuantizedLinear layer and adds Gaussian noise to the
    dequantized weights during each forward pass. The noise standard deviation
    is computed from the weight values using a configurable function.
    
    Attributes:
        base_layer: The wrapped QuantizedLinear layer
        noise_fn: Function that computes sigma from weights
        function_params: Parameters for the noise function
        enabled: Whether noise injection is currently enabled
    """
    
    def __init__(
        self,
        base_layer: QuantizedLinear,
        noise_fn: NoiseFunctionType,
        function_params: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
    ):
        """Initialize the noisy wrapper.
        
        Args:
            base_layer: QuantizedLinear layer to wrap
            noise_fn: Function that computes sigma from weights
            function_params: Parameters for the noise function
            enabled: Whether to initially enable noise injection
        """
        super().__init__()
        
        self.base_layer = base_layer
        self.noise_fn = noise_fn
        self.function_params = function_params or {}
        self.enabled = enabled
        
        # Copy attributes from base layer for compatibility
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.bits = base_layer.bits
        self.group_size = base_layer.group_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional noise injection.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Dequantize weights
        weight = self.base_layer._dequantize_weight()
        weight = weight[:, :self.in_features]  # Trim padding
        
        # Apply noise if enabled and in training mode or always enabled
        if self.enabled:
            # Compute sigma for each weight
            sigma = self.noise_fn(weight, **self.function_params)
            
            # Sample and add noise
            noise = torch.randn_like(weight) * sigma
            weight = weight + noise
        
        # Match dtype with input
        weight = weight.to(x.dtype)
        
        # Linear operation
        output = torch.nn.functional.linear(x, weight, self.base_layer.bias)
        
        return output
    
    def enable_noise(self) -> None:
        """Enable noise injection."""
        self.enabled = True
    
    def disable_noise(self) -> None:
        """Disable noise injection."""
        self.enabled = False
    
    def set_noise_params(self, **params) -> None:
        """Update noise function parameters.
        
        Args:
            **params: New parameters for the noise function
        """
        self.function_params.update(params)


class NoisyLinear(nn.Module):
    """A wrapper around nn.Linear that adds noise at inference time.
    
    Similar to NoisyQuantizedLinear but for regular Linear layers.
    """
    
    def __init__(
        self,
        base_layer: nn.Linear,
        noise_fn: NoiseFunctionType,
        function_params: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
    ):
        """Initialize the noisy wrapper.
        
        Args:
            base_layer: Linear layer to wrap
            noise_fn: Function that computes sigma from weights
            function_params: Parameters for the noise function
            enabled: Whether to initially enable noise injection
        """
        super().__init__()
        
        self.base_layer = base_layer
        self.noise_fn = noise_fn
        self.function_params = function_params or {}
        self.enabled = enabled
        
        # Copy attributes from base layer for compatibility
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional noise injection.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        weight = self.base_layer.weight
        
        if self.enabled:
            # Compute sigma for each weight
            sigma = self.noise_fn(weight, **self.function_params)
            
            # Sample and add noise
            noise = torch.randn_like(weight) * sigma
            weight = weight + noise
        
        # Linear operation
        output = torch.nn.functional.linear(x, weight, self.base_layer.bias)
        
        return output
    
    def enable_noise(self) -> None:
        """Enable noise injection."""
        self.enabled = True
    
    def disable_noise(self) -> None:
        """Disable noise injection."""
        self.enabled = False
    
    def set_noise_params(self, **params) -> None:
        """Update noise function parameters.
        
        Args:
            **params: New parameters for the noise function
        """
        self.function_params.update(params)


class DynamicNoiseWrapper:
    """Wraps model layers to apply dynamic noise at inference time.
    
    This class replaces QuantizedLinear (and optionally Linear) layers with
    noisy wrappers that add fresh Gaussian noise during each forward pass.
    
    Example:
        >>> wrapper = DynamicNoiseWrapper(
        ...     noise_function="proportional",
        ...     function_params={"scale": 0.05},
        ... )
        >>> model = wrapper.apply(model)
        >>> 
        >>> # Later, disable noise for clean inference:
        >>> wrapper.disable_noise()
    """
    
    def __init__(
        self,
        noise_function: Union[str, NoiseFunctionType] = "proportional",
        function_params: Optional[Dict[str, Any]] = None,
        target_layers: Optional[List[str]] = None,
        wrap_linear: bool = False,
    ):
        """Initialize the dynamic noise wrapper.
        
        Args:
            noise_function: Name of registered noise function or a callable
            function_params: Parameters to pass to the noise function
            target_layers: Optional list of layer name patterns to target
            wrap_linear: Whether to also wrap regular Linear layers
        """
        if isinstance(noise_function, str):
            self.noise_fn = get_noise_function(noise_function)
            self.noise_fn_name = noise_function
        else:
            self.noise_fn = noise_function
            self.noise_fn_name = getattr(noise_function, "__name__", "custom")
        
        self.function_params = function_params or {}
        self.target_layers = target_layers
        self.wrap_linear = wrap_linear
        
        # Track wrapped layers for later control
        self._wrapped_layers: Dict[str, nn.Module] = {}
        
        logger.info(f"Initialized DynamicNoiseWrapper")
        logger.info(f"  function={self.noise_fn_name}, params={self.function_params}")
    
    def apply(
        self,
        model: Union[PreTrainedModel, ModelWrapper],
    ) -> Union[PreTrainedModel, ModelWrapper]:
        """Wrap model layers with noisy versions.
        
        Args:
            model: The model to wrap
            
        Returns:
            The model with wrapped layers
        """
        if isinstance(model, ModelWrapper):
            hf_model = model.model
        else:
            hf_model = model
        
        # Collect layers to wrap
        layers_to_wrap = []
        for name, module in hf_model.named_modules():
            if self._should_wrap_layer(name, module):
                layers_to_wrap.append((name, module))
        
        # Wrap each layer
        for name, module in layers_to_wrap:
            self._wrap_layer(hf_model, name, module)
        
        logger.info(f"Wrapped {len(layers_to_wrap)} layers with dynamic noise")
        
        return model
    
    def _should_wrap_layer(self, name: str, module: nn.Module) -> bool:
        """Check if a layer should be wrapped.
        
        Args:
            name: Layer name
            module: Layer module
            
        Returns:
            True if layer should be wrapped
        """
        # Check layer type
        if isinstance(module, QuantizedLinear):
            pass  # Always consider quantized layers
        elif isinstance(module, nn.Linear) and self.wrap_linear:
            pass  # Consider linear layers if enabled
        else:
            return False
        
        # Skip already wrapped layers
        if isinstance(module, (NoisyQuantizedLinear, NoisyLinear)):
            return False
        
        # If no target patterns, wrap all eligible layers
        if self.target_layers is None:
            return True
        
        # Check if name matches any target pattern
        return any(pattern in name for pattern in self.target_layers)
    
    def _wrap_layer(
        self,
        model: nn.Module,
        name: str,
        layer: nn.Module,
    ) -> None:
        """Wrap a single layer with its noisy version.
        
        Args:
            model: Parent model
            name: Dotted name of the layer
            layer: Layer to wrap
        """
        # Create noisy wrapper
        if isinstance(layer, QuantizedLinear):
            noisy_layer = NoisyQuantizedLinear(
                layer,
                self.noise_fn,
                self.function_params.copy(),
            )
        else:
            noisy_layer = NoisyLinear(
                layer,
                self.noise_fn,
                self.function_params.copy(),
            )
        
        # Replace the layer
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        
        setattr(parent, parts[-1], noisy_layer)
        
        # Track the wrapped layer
        self._wrapped_layers[name] = noisy_layer
        
        logger.debug(f"  Wrapped: {name}")
    
    def enable_noise(self) -> None:
        """Enable noise for all wrapped layers."""
        for layer in self._wrapped_layers.values():
            layer.enable_noise()
        logger.info("Enabled dynamic noise for all wrapped layers")
    
    def disable_noise(self) -> None:
        """Disable noise for all wrapped layers."""
        for layer in self._wrapped_layers.values():
            layer.disable_noise()
        logger.info("Disabled dynamic noise for all wrapped layers")
    
    def set_noise_params(self, **params) -> None:
        """Update noise parameters for all wrapped layers.
        
        Args:
            **params: New parameters for the noise function
        """
        self.function_params.update(params)
        for layer in self._wrapped_layers.values():
            layer.set_noise_params(**params)
        logger.info(f"Updated noise params: {params}")
    
    def get_wrapped_layers(self) -> Dict[str, nn.Module]:
        """Get all wrapped layers.
        
        Returns:
            Dictionary mapping layer names to wrapped modules
        """
        return self._wrapped_layers.copy()

"""Noise function registry with built-in functions for computing noise standard deviation.

This module provides a registry for noise functions that compute the standard deviation
of Gaussian noise to be applied to quantized weights. Users can register custom functions
via decorators.

Example:
    >>> from analog_ptq.noise.functions import register_noise_function, get_noise_function
    >>> 
    >>> @register_noise_function("my_custom")
    >>> def my_custom_sigma(weight: torch.Tensor, scale: float = 0.1) -> torch.Tensor:
    >>>     return scale * torch.sqrt(weight.abs())
    >>> 
    >>> # Later, retrieve and use:
    >>> sigma_fn = get_noise_function("my_custom")
    >>> sigma = sigma_fn(weights, scale=0.1)
"""

from typing import Any, Callable, Dict, Optional

import torch

from analog_ptq.utils.logging import get_logger


logger = get_logger(__name__)


# Type alias for noise functions
# Noise functions take a weight tensor and keyword arguments, returning a sigma tensor
NoiseFunctionType = Callable[..., torch.Tensor]


class NoiseFunctionRegistry:
    """Registry for noise standard deviation functions.
    
    Allows registration of custom functions that compute sigma = f(weight)
    for Gaussian noise injection.
    """
    
    def __init__(self):
        """Initialize the registry."""
        self._functions: Dict[str, NoiseFunctionType] = {}
    
    def register(self, name: str) -> Callable[[NoiseFunctionType], NoiseFunctionType]:
        """Register a noise function.
        
        Can be used as a decorator:
        
            @registry.register("my_function")
            def my_function(weight: torch.Tensor, **kwargs) -> torch.Tensor:
                ...
        
        Args:
            name: Name to register the function under
            
        Returns:
            Decorator function
        """
        def decorator(fn: NoiseFunctionType) -> NoiseFunctionType:
            if name in self._functions:
                logger.warning(f"Overwriting existing noise function: {name}")
            self._functions[name] = fn
            logger.debug(f"Registered noise function: {name}")
            return fn
        
        return decorator
    
    def get(self, name: str) -> NoiseFunctionType:
        """Get a registered noise function by name.
        
        Args:
            name: Function name
            
        Returns:
            The noise function
            
        Raises:
            KeyError: If function is not registered
        """
        if name not in self._functions:
            available = list(self._functions.keys())
            raise KeyError(
                f"Unknown noise function: {name}. Available: {available}"
            )
        return self._functions[name]
    
    def list_functions(self) -> list:
        """List all registered function names.
        
        Returns:
            List of function names
        """
        return list(self._functions.keys())
    
    def __contains__(self, name: str) -> bool:
        """Check if a function is registered."""
        return name in self._functions


# Global registry instance
_registry = NoiseFunctionRegistry()


def register_noise_function(name: str) -> Callable[[NoiseFunctionType], NoiseFunctionType]:
    """Register a noise function with the global registry.
    
    Use as a decorator:
    
        @register_noise_function("my_custom")
        def my_custom_sigma(weight: torch.Tensor, scale: float = 0.1) -> torch.Tensor:
            return scale * torch.sqrt(weight.abs())
    
    Args:
        name: Name to register the function under
        
    Returns:
        Decorator function
    """
    return _registry.register(name)


def get_noise_function(name: str) -> NoiseFunctionType:
    """Get a noise function from the global registry.
    
    Args:
        name: Function name
        
    Returns:
        The noise function
    """
    return _registry.get(name)


def list_noise_functions() -> list:
    """List all registered noise functions.
    
    Returns:
        List of function names
    """
    return _registry.list_functions()


# =============================================================================
# Built-in Noise Functions
# =============================================================================


@register_noise_function("constant")
def constant_sigma(weight: torch.Tensor, scale: float = 0.1) -> torch.Tensor:
    """Constant standard deviation for all weights.
    
    sigma = scale (same value for all weights)
    
    Args:
        weight: Weight tensor (used only for shape/device)
        scale: The constant standard deviation value
        
    Returns:
        Tensor of constant sigma values with same shape as weight
    """
    return torch.full_like(weight, scale, dtype=torch.float32)


@register_noise_function("proportional")
def proportional_sigma(weight: torch.Tensor, scale: float = 0.1) -> torch.Tensor:
    """Standard deviation proportional to absolute weight value.
    
    sigma = scale * |weight|
    
    Args:
        weight: Weight tensor
        scale: Scaling factor for the proportional relationship
        
    Returns:
        Tensor of sigma values
    """
    return scale * weight.abs().float()


@register_noise_function("polynomial")
def polynomial_sigma(
    weight: torch.Tensor,
    scale: float = 0.1,
    power: float = 1.0,
) -> torch.Tensor:
    """Standard deviation as polynomial function of absolute weight value.
    
    sigma = scale * |weight|^power
    
    Args:
        weight: Weight tensor
        scale: Scaling factor
        power: Polynomial power (exponent)
        
    Returns:
        Tensor of sigma values
    """
    return scale * torch.pow(weight.abs().float(), power)


@register_noise_function("sqrt")
def sqrt_sigma(weight: torch.Tensor, scale: float = 0.1) -> torch.Tensor:
    """Standard deviation proportional to square root of absolute weight value.
    
    sigma = scale * sqrt(|weight|)
    
    This is a common choice in analog computing noise models.
    
    Args:
        weight: Weight tensor
        scale: Scaling factor
        
    Returns:
        Tensor of sigma values
    """
    return scale * torch.sqrt(weight.abs().float())


@register_noise_function("relative")
def relative_sigma(
    weight: torch.Tensor,
    scale: float = 0.1,
    min_sigma: float = 1e-6,
) -> torch.Tensor:
    """Relative noise with minimum floor.
    
    sigma = max(scale * |weight|, min_sigma)
    
    This ensures a minimum noise level even for near-zero weights.
    
    Args:
        weight: Weight tensor
        scale: Scaling factor for relative noise
        min_sigma: Minimum sigma value (floor)
        
    Returns:
        Tensor of sigma values
    """
    sigma = scale * weight.abs().float()
    return torch.clamp(sigma, min=min_sigma)


@register_noise_function("affine")
def affine_sigma(
    weight: torch.Tensor,
    scale: float = 0.1,
    offset: float = 0.01,
) -> torch.Tensor:
    """Affine function of absolute weight value.
    
    sigma = scale * |weight| + offset
    
    Combines proportional noise with a constant baseline.
    
    Args:
        weight: Weight tensor
        scale: Scaling factor for proportional component
        offset: Constant offset (baseline noise)
        
    Returns:
        Tensor of sigma values
    """
    return scale * weight.abs().float() + offset

"""Noise injection module for analog computing simulation.

This module provides tools for injecting Gaussian noise into quantized model weights
to simulate analog computing hardware characteristics.

Two modes of noise injection are supported:

1. **Static noise** (via WeightNoiseInjector):
   Applied once after quantization, permanently modifying the weights.
   Useful for simulating fixed hardware variations.

2. **Dynamic noise** (via DynamicNoiseWrapper):
   Applied at inference time during each forward pass.
   Useful for simulating runtime analog noise.

The noise standard deviation can be a configurable function of the weight values,
allowing simulation of various analog noise models.

Example:
    Static noise injection:
    
    >>> from analog_ptq.noise import WeightNoiseInjector
    >>> injector = WeightNoiseInjector(
    ...     noise_function="proportional",
    ...     function_params={"scale": 0.05},
    ...     seed=42,
    ... )
    >>> model = injector.apply(quantized_model)
    
    Dynamic noise injection:
    
    >>> from analog_ptq.noise import DynamicNoiseWrapper
    >>> wrapper = DynamicNoiseWrapper(
    ...     noise_function="proportional",
    ...     function_params={"scale": 0.05},
    ... )
    >>> model = wrapper.apply(quantized_model)
    >>> wrapper.disable_noise()  # Disable for clean inference
    
    Custom noise function:
    
    >>> from analog_ptq.noise import register_noise_function
    >>> 
    >>> @register_noise_function("my_custom")
    >>> def my_custom_sigma(weight, scale=0.1, power=0.5):
    >>>     return scale * torch.pow(weight.abs(), power)
"""

from analog_ptq.noise.functions import (
    NoiseFunctionType,
    get_noise_function,
    list_noise_functions,
    register_noise_function,
    # Built-in functions
    constant_sigma,
    proportional_sigma,
    polynomial_sigma,
    sqrt_sigma,
    relative_sigma,
    affine_sigma,
)
from analog_ptq.noise.injector import WeightNoiseInjector
from analog_ptq.noise.dynamic import (
    DynamicNoiseWrapper,
    NoisyQuantizedLinear,
    NoisyLinear,
)


__all__ = [
    # Registry functions
    "register_noise_function",
    "get_noise_function",
    "list_noise_functions",
    "NoiseFunctionType",
    # Static noise injection
    "WeightNoiseInjector",
    # Dynamic noise injection
    "DynamicNoiseWrapper",
    "NoisyQuantizedLinear",
    "NoisyLinear",
    # Built-in noise functions
    "constant_sigma",
    "proportional_sigma",
    "polynomial_sigma",
    "sqrt_sigma",
    "relative_sigma",
    "affine_sigma",
]

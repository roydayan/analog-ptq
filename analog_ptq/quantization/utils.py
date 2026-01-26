"""Quantization utility functions."""

from typing import Optional, Tuple

import torch
import torch.nn as nn


def quantize_tensor(
    tensor: torch.Tensor,
    bits: int = 4,
    group_size: int = 128,
    symmetric: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Quantize a tensor to specified bit width.
    
    Args:
        tensor: Input tensor to quantize
        bits: Number of bits for quantization
        group_size: Group size for group-wise quantization (-1 for per-channel)
        symmetric: Whether to use symmetric quantization
        
    Returns:
        Tuple of (quantized_tensor, scales, zeros)
        zeros is None for symmetric quantization
    """
    original_shape = tensor.shape
    
    # Reshape for group-wise quantization
    if group_size > 0 and tensor.numel() > group_size:
        # Reshape to [..., num_groups, group_size]
        if tensor.dim() == 2:
            out_features, in_features = tensor.shape
            num_groups = (in_features + group_size - 1) // group_size
            padded_size = num_groups * group_size
            
            # Pad if necessary
            if padded_size > in_features:
                tensor = torch.nn.functional.pad(
                    tensor, (0, padded_size - in_features)
                )
            
            tensor = tensor.view(out_features, num_groups, group_size)
            reduce_dims = (2,)
        else:
            reduce_dims = (-1,)
    else:
        reduce_dims = tuple(range(tensor.dim()))
    
    # Compute quantization parameters
    if symmetric:
        max_val = tensor.abs().amax(dim=reduce_dims, keepdim=True)
        scales = max_val / (2 ** (bits - 1) - 1)
        scales = scales.clamp(min=1e-8)
        zeros = None
        
        # Quantize
        quantized = torch.round(tensor / scales)
        quantized = quantized.clamp(-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
    else:
        min_val = tensor.amin(dim=reduce_dims, keepdim=True)
        max_val = tensor.amax(dim=reduce_dims, keepdim=True)
        
        scales = (max_val - min_val) / (2 ** bits - 1)
        scales = scales.clamp(min=1e-8)
        zeros = torch.round(-min_val / scales)
        
        # Quantize
        quantized = torch.round(tensor / scales + zeros)
        quantized = quantized.clamp(0, 2 ** bits - 1)
    
    return quantized, scales, zeros


def dequantize_tensor(
    quantized: torch.Tensor,
    scales: torch.Tensor,
    zeros: Optional[torch.Tensor] = None,
    original_shape: Optional[Tuple[int, ...]] = None,
) -> torch.Tensor:
    """Dequantize a tensor back to floating point.
    
    Args:
        quantized: Quantized tensor
        scales: Scale factors
        zeros: Zero points (None for symmetric)
        original_shape: Original tensor shape (for reshaping)
        
    Returns:
        Dequantized tensor
    """
    if zeros is not None:
        dequantized = (quantized - zeros) * scales
    else:
        dequantized = quantized * scales
    
    if original_shape is not None:
        # Reshape and trim padding
        dequantized = dequantized.view(-1)[:torch.tensor(original_shape).prod().item()]
        dequantized = dequantized.view(original_shape)
    
    return dequantized


def compute_quantization_error(
    original: torch.Tensor,
    quantized: torch.Tensor,
    scales: torch.Tensor,
    zeros: Optional[torch.Tensor] = None,
) -> dict:
    """Compute quantization error metrics.
    
    Args:
        original: Original tensor
        quantized: Quantized tensor
        scales: Scale factors
        zeros: Zero points
        
    Returns:
        Dictionary with error metrics (mse, mae, max_error)
    """
    dequantized = dequantize_tensor(quantized, scales, zeros, original.shape)
    
    error = original - dequantized
    
    return {
        "mse": (error ** 2).mean().item(),
        "mae": error.abs().mean().item(),
        "max_error": error.abs().max().item(),
    }


def find_linear_layers(
    module: nn.Module,
    prefix: str = "",
    skip_patterns: Optional[list] = None,
) -> dict:
    """Find all linear layers in a module.
    
    Args:
        module: Module to search
        prefix: Prefix for layer names
        skip_patterns: List of name patterns to skip
        
    Returns:
        Dictionary mapping names to Linear modules
    """
    skip_patterns = skip_patterns or ["lm_head"]
    layers = {}
    
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        
        # Check if should skip
        should_skip = any(pattern in full_name for pattern in skip_patterns)
        
        if isinstance(child, nn.Linear) and not should_skip:
            layers[full_name] = child
        else:
            # Recurse
            layers.update(find_linear_layers(child, full_name, skip_patterns))
    
    return layers


class QuantizedLinear(nn.Module):
    """A quantized linear layer for inference.
    
    Stores weights in quantized format and dequantizes during forward pass.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 4,
        group_size: int = 128,
        bias: bool = True,
    ):
        """Initialize quantized linear layer.
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            bits: Quantization bit width
            group_size: Group size for group-wise quantization
            bias: Whether to include bias
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size
        
        # Calculate number of groups
        self.num_groups = (in_features + group_size - 1) // group_size
        
        # Register buffers for quantized weights
        self.register_buffer(
            "qweight",
            torch.zeros(out_features, self.num_groups * group_size, dtype=torch.int8)
        )
        self.register_buffer(
            "scales",
            torch.zeros(out_features, self.num_groups, dtype=torch.float16)
        )
        self.register_buffer(
            "zeros",
            torch.zeros(out_features, self.num_groups, dtype=torch.float16)
        )
        
        if bias:
            self.register_buffer("bias", torch.zeros(out_features))
        else:
            self.register_buffer("bias", None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dequantization.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Dequantize weights
        weight = self._dequantize_weight()
        
        # Trim to original size
        weight = weight[:, :self.in_features]
        
        # Match dtype with input to avoid dtype mismatch errors
        weight = weight.to(x.dtype)
        
        # Linear operation
        output = torch.nn.functional.linear(x, weight, self.bias)
        
        return output
    
    def _dequantize_weight(self) -> torch.Tensor:
        """Dequantize the stored weights.
        
        Returns:
            Dequantized weight tensor
        """
        # Reshape for broadcasting
        qweight = self.qweight.view(self.out_features, self.num_groups, self.group_size)
        scales = self.scales.unsqueeze(-1)
        zeros = self.zeros.unsqueeze(-1)
        
        # Dequantize
        weight = (qweight.float() - zeros) * scales
        
        # Reshape back
        weight = weight.view(self.out_features, -1)
        
        return weight
    
    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        bits: int = 4,
        group_size: int = 128,
    ) -> "QuantizedLinear":
        """Create a QuantizedLinear from a regular Linear layer.
        
        Args:
            linear: Source Linear layer
            bits: Quantization bit width
            group_size: Group size
            
        Returns:
            New QuantizedLinear layer
        """
        qlayer = cls(
            linear.in_features,
            linear.out_features,
            bits=bits,
            group_size=group_size,
            bias=linear.bias is not None,
        )
        
        # Quantize weights
        weight = linear.weight.data
        qweight, scales, zeros = quantize_tensor(
            weight, bits=bits, group_size=group_size, symmetric=False
        )
        
        qlayer.qweight.copy_(qweight.view(qlayer.qweight.shape).to(torch.int8))
        qlayer.scales.copy_(scales.view(qlayer.scales.shape).to(torch.float16))
        if zeros is not None:
            qlayer.zeros.copy_(zeros.view(qlayer.zeros.shape).to(torch.float16))
        
        if linear.bias is not None:
            qlayer.bias.copy_(linear.bias.data)
        
        return qlayer

"""Sigma (noise standard deviation) models for NA-GPTQ.

This module provides models for computing the noise variance σ²(z) at each
quantization level z. These are used in the noise-aware quantization objective:

    J(ŵ) = (w - ŵ)ᵀ H (w - ŵ) + Σᵢ aᵢ σ²(ŵᵢ)

where aᵢ = Hᵢᵢ is the diagonal of the Hessian.

Supported models:
    - constant: σ²(z) = σ₀²
    - affine: σ(z) = σ₀ + α|z|, so σ²(z) = (σ₀ + α|z|)²
    - power: σ(z) = σ₀ + α(|z|/z_max)^p
    - lookup: σ²(z_q) from a pre-defined table
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Union

import torch


class SigmaModel(ABC):
    """Abstract base class for noise variance models.
    
    Subclasses must implement `sigma_squared()` which returns σ²(z) for
    each quantization level z in the grid.
    """
    
    @abstractmethod
    def sigma_squared(self, z: torch.Tensor) -> torch.Tensor:
        """Compute σ²(z) for quantization levels.
        
        Args:
            z: Tensor of quantization levels (any shape)
            
        Returns:
            Tensor of same shape containing σ²(z) values
        """
        pass
    
    def sigma(self, z: torch.Tensor) -> torch.Tensor:
        """Compute σ(z) = sqrt(σ²(z)).
        
        Args:
            z: Tensor of quantization levels
            
        Returns:
            Tensor of σ(z) values
        """
        return torch.sqrt(self.sigma_squared(z))


class ConstantSigmaModel(SigmaModel):
    """Constant noise model: σ(z) = σ₀ for all z.
    
    This is the simplest model where noise variance is independent of
    the quantization level.
    
    Attributes:
        sigma0: The constant noise standard deviation
    """
    
    def __init__(self, sigma0: float = 0.01):
        """Initialize constant sigma model.
        
        Args:
            sigma0: Constant noise standard deviation (σ₀ ≥ 0)
        """
        if sigma0 < 0:
            raise ValueError(f"sigma0 must be non-negative, got {sigma0}")
        self.sigma0 = sigma0
        self._sigma0_squared = sigma0 ** 2
    
    def sigma_squared(self, z: torch.Tensor) -> torch.Tensor:
        """Return constant σ₀² for all levels.
        
        Args:
            z: Tensor of quantization levels (used only for shape/device)
            
        Returns:
            Tensor filled with σ₀²
        """
        return torch.full_like(z, self._sigma0_squared, dtype=torch.float32)
    
    def __repr__(self) -> str:
        return f"ConstantSigmaModel(sigma0={self.sigma0})"


class AffineSigmaModel(SigmaModel):
    """Affine magnitude noise model: σ(z) = σ₀ + α|z|.
    
    This models noise that increases linearly with the magnitude of the
    quantized value, which is common in analog computing where larger
    conductance values have proportionally larger variations.
    
    The variance is: σ²(z) = (σ₀ + α|z|)²
    
    Attributes:
        sigma0: Base noise standard deviation (intercept)
        alpha: Slope of linear dependence on |z|
    """
    
    def __init__(self, sigma0: float = 0.01, alpha: float = 0.05):
        """Initialize affine sigma model.
        
        Args:
            sigma0: Base noise std (σ₀ ≥ 0)
            alpha: Magnitude coefficient (α ≥ 0)
        """
        if sigma0 < 0:
            raise ValueError(f"sigma0 must be non-negative, got {sigma0}")
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        self.sigma0 = sigma0
        self.alpha = alpha
    
    def sigma_squared(self, z: torch.Tensor) -> torch.Tensor:
        """Compute σ²(z) = (σ₀ + α|z|)².
        
        Args:
            z: Tensor of quantization levels
            
        Returns:
            Tensor of σ²(z) values
        """
        sigma = self.sigma0 + self.alpha * z.abs().float()
        return sigma ** 2
    
    def __repr__(self) -> str:
        return f"AffineSigmaModel(sigma0={self.sigma0}, alpha={self.alpha})"


class PowerSigmaModel(SigmaModel):
    """Power-law magnitude noise model: σ(z) = σ₀ + α(|z|/z_max)^p.
    
    This provides more flexibility than affine by allowing sub-linear (p < 1)
    or super-linear (p > 1) dependence on magnitude.
    
    Attributes:
        sigma0: Base noise standard deviation
        alpha: Scale coefficient
        power: Exponent for magnitude dependence
        z_max: Maximum |z| for normalization (typically max grid value)
    """
    
    def __init__(
        self,
        sigma0: float = 0.01,
        alpha: float = 0.05,
        power: float = 1.0,
        z_max: float = 1.0,
    ):
        """Initialize power sigma model.
        
        Args:
            sigma0: Base noise std (σ₀ ≥ 0)
            alpha: Scale coefficient (α ≥ 0)
            power: Exponent p (typically > 0)
            z_max: Normalization factor (z_max > 0)
        """
        if sigma0 < 0:
            raise ValueError(f"sigma0 must be non-negative, got {sigma0}")
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        if z_max <= 0:
            raise ValueError(f"z_max must be positive, got {z_max}")
        
        self.sigma0 = sigma0
        self.alpha = alpha
        self.power = power
        self.z_max = z_max
    
    def sigma_squared(self, z: torch.Tensor) -> torch.Tensor:
        """Compute σ²(z) = (σ₀ + α(|z|/z_max)^p)².
        
        Args:
            z: Tensor of quantization levels
            
        Returns:
            Tensor of σ²(z) values
        """
        normalized = z.abs().float() / self.z_max
        sigma = self.sigma0 + self.alpha * torch.pow(normalized, self.power)
        return sigma ** 2
    
    def __repr__(self) -> str:
        return (f"PowerSigmaModel(sigma0={self.sigma0}, alpha={self.alpha}, "
                f"power={self.power}, z_max={self.z_max})")


class LookupTableSigmaModel(SigmaModel):
    """Per-level lookup table for σ²(z_q).
    
    This allows arbitrary noise profiles by specifying σ² for each
    quantization level directly. Useful when noise characteristics
    are measured empirically from hardware.
    
    The lookup table maps integer quantization indices to variance values.
    For symmetric quantization with b bits, indices are in [-2^(b-1)+1, 2^(b-1)-1].
    
    Attributes:
        sigma2_table: Tensor mapping level index to σ² value
        bits: Bit width of quantization
        q_min: Minimum quantization index
        q_max: Maximum quantization index
    """
    
    def __init__(
        self,
        sigma2_table: torch.Tensor,
        bits: int = 4,
        scale: float = 1.0,
    ):
        """Initialize lookup table sigma model.
        
        Args:
            sigma2_table: 1D tensor of σ² values for each level.
                         Length should be 2^bits - 1 for symmetric quantization
                         (levels from -Qmax to +Qmax where Qmax = 2^(b-1) - 1).
            bits: Quantization bit width
            scale: Scale factor to convert z to grid index (Δ in z = Δ * q)
        """
        self.bits = bits
        self.scale = scale
        self.q_max = 2 ** (bits - 1) - 1
        self.q_min = -self.q_max
        
        expected_len = 2 * self.q_max + 1
        if sigma2_table.numel() != expected_len:
            raise ValueError(
                f"sigma2_table must have {expected_len} entries for {bits}-bit "
                f"symmetric quantization, got {sigma2_table.numel()}"
            )
        
        # Store as buffer (will be moved with model)
        self.sigma2_table = sigma2_table.float()
        self._offset = self.q_max  # To convert q to table index: idx = q + offset
    
    def sigma_squared(self, z: torch.Tensor) -> torch.Tensor:
        """Look up σ²(z) from the table.
        
        Args:
            z: Tensor of quantization levels (scaled values, z = scale * q)
            
        Returns:
            Tensor of σ²(z) values from lookup table
        """
        # Convert z back to integer indices
        q = torch.round(z / self.scale).long()
        q = torch.clamp(q, self.q_min, self.q_max)
        
        # Look up in table
        indices = q + self._offset
        
        # Move table to same device as z
        table = self.sigma2_table.to(z.device)
        
        return table[indices]
    
    def __repr__(self) -> str:
        return f"LookupTableSigmaModel(bits={self.bits}, scale={self.scale})"


def create_sigma_model(
    model_type: str,
    params: Optional[Dict] = None,
    sigma2_table: Optional[torch.Tensor] = None,
    bits: int = 4,
    scale: float = 1.0,
) -> SigmaModel:
    """Factory function to create a sigma model.
    
    Args:
        model_type: One of "constant", "affine", "power", "lookup"
        params: Dictionary of model parameters. Keys depend on model_type:
                - constant: {"sigma0": float}
                - affine: {"sigma0": float, "alpha": float}
                - power: {"sigma0": float, "alpha": float, "power": float, "z_max": float}
        sigma2_table: Required for "lookup" type
        bits: Bit width (used for lookup table)
        scale: Scale factor (used for lookup table)
        
    Returns:
        Configured SigmaModel instance
        
    Example:
        >>> model = create_sigma_model("affine", {"sigma0": 0.01, "alpha": 0.05})
        >>> sigma2 = model.sigma_squared(torch.tensor([0.0, 1.0, 2.0]))
    """
    params = params or {}
    
    if model_type == "constant":
        return ConstantSigmaModel(
            sigma0=params.get("sigma0", 0.01),
        )
    
    elif model_type == "affine":
        return AffineSigmaModel(
            sigma0=params.get("sigma0", 0.01),
            alpha=params.get("alpha", 0.05),
        )
    
    elif model_type == "power":
        # Compute z_max from bits and scale if not provided
        q_max = 2 ** (bits - 1) - 1
        default_z_max = scale * q_max
        
        return PowerSigmaModel(
            sigma0=params.get("sigma0", 0.01),
            alpha=params.get("alpha", 0.05),
            power=params.get("power", 1.0),
            z_max=params.get("z_max", default_z_max),
        )
    
    elif model_type == "lookup":
        if sigma2_table is None:
            raise ValueError("sigma2_table is required for lookup model")
        return LookupTableSigmaModel(
            sigma2_table=sigma2_table,
            bits=bits,
            scale=scale,
        )
    
    else:
        raise ValueError(
            f"Unknown sigma model type: {model_type}. "
            f"Supported: constant, affine, power, lookup"
        )

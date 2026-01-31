"""Noise-Aware GPTQ (NA-GPTQ) quantization implementation.

Performance logging: Set environment variable NAGPTQ_DEBUG=1 to enable detailed timing logs.

This module implements NA-GPTQ, which extends GPTQ to be aware of deployment-time
analog noise. It optimizes the expected reconstruction error including noise:

    J(ŵ) = (w - ŵ)ᵀ H (w - ŵ) + Σᵢ aᵢ σ²(ŵᵢ)

where:
    - H = XXᵀ + λI is the Hessian approximation (X is d×N calibration input)
    - aᵢ = Hᵢᵢ is the diagonal (sensitivity to noise)
    - σ²(ŵ) is the noise variance at quantized weight ŵ

Key modifications from standard GPTQ:
    1. Noise-aware rounding: argmin_z [(z - u)² / sensitivity + aᵢ σ²(z)]
       where sensitivity = Hinv[i,i] (same as GPTQ uses for error correction)
    2. Noise-aware update: (H_FF + 0.5D)δ = H_FS Δ_S - 0.5g
       where g_j = aⱼ ρ'(wⱼ), D_jj = aⱼ ρ''(wⱼ) (D=0 if use_second_derivative=False)
       
    ρ(u) is the soft-quantization expected noise variance:
        ρ(u) = Σ_q p_q(u) σ²(z_q)
        p_q(u) = exp(-(u - z_q)² / (2τ²)) / Z
    
    Quantization grid: z_q = Δ·q for q ∈ [-Qmax, Qmax], Qmax = 2^(b-1) - 1
    τ is interpreted as relative to step size Δ: τ_eff = tau * Δ

References:
    - GPTQ paper: https://arxiv.org/abs/2210.17323
    - NA-GPTQ extends GPTQ with noise-aware optimization
"""

import os
import time
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from analog_ptq.models.wrapper import ModelWrapper
from analog_ptq.quantization.gptq import GPTQQuantizer
from analog_ptq.quantization.sigma_models import SigmaModel, create_sigma_model
from analog_ptq.quantization.utils import QuantizedLinear
from analog_ptq.utils.logging import get_logger


logger = get_logger(__name__)

# Enable detailed timing logs with NAGPTQ_DEBUG=1
_DEBUG = os.environ.get("NAGPTQ_DEBUG", "0") == "1"


class Timer:
    """Simple context manager for timing code blocks."""
    
    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled
        self.elapsed = 0.0
    
    def __enter__(self):
        if self.enabled:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            self.start = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        if self.enabled:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            self.elapsed = time.perf_counter() - self.start
            logger.info(f"  [TIMER] {self.name}: {self.elapsed:.3f}s")


# =============================================================================
# Utility Functions
# =============================================================================


def build_uniform_symmetric_grid(
    bits: int,
    scale: float,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Build a uniform symmetric quantization grid.
    
    Creates grid z_q = Δ * q for q ∈ [-Qmax, ..., Qmax] where Qmax = 2^(b-1) - 1.
    This is the standard symmetric integer quantization grid.
    
    Args:
        bits: Bit width (e.g., 4 for int4)
        scale: Step size Δ (quantization scale)
        device: Target device for the tensor
        
    Returns:
        1D tensor of quantization levels, shape [2*Qmax + 1]
        
    Example:
        >>> grid = build_uniform_symmetric_grid(bits=4, scale=0.1)
        >>> # Returns [-0.7, -0.6, ..., 0.0, ..., 0.6, 0.7] for Qmax=7
    """
    q_max = 2 ** (bits - 1) - 1
    q_values = torch.arange(-q_max, q_max + 1, dtype=torch.float32, device=device)
    return scale * q_values


def compute_soft_assign(
    u: torch.Tensor,
    grid: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """Compute soft assignment probabilities over quantization grid.
    
    Computes softmax probabilities:
        p_q(u) = exp(-(u - z_q)² / (2τ²)) / Z
    
    where Z = Σ_r exp(-(u - z_r)² / (2τ²)) is the partition function.
    
    As τ → 0, this approaches hard assignment (one-hot at nearest grid point).
    As τ → ∞, this approaches uniform distribution.
    
    Args:
        u: Input values, shape [...] (any shape)
        grid: Quantization grid, shape [G] where G = 2*Qmax + 1
        tau: Temperature parameter (τ > 0)
        
    Returns:
        Soft assignment probabilities, shape [..., G]
        
    Note:
        Uses numerically stable softmax with log-sum-exp trick.
    """
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    
    # Expand dimensions for broadcasting: u[..., None] - grid[None, ...]
    # u: [...] -> [..., 1]
    # grid: [G] -> [1, ..., 1, G] (broadcast)
    u_expanded = u.unsqueeze(-1)  # [..., 1]
    
    # Compute negative squared distances / (2τ²)
    neg_dist_sq = -((u_expanded - grid) ** 2) / (2 * tau ** 2)
    
    # Softmax for numerical stability
    p = torch.softmax(neg_dist_sq, dim=-1)
    
    return p


class RhoDerivatives(NamedTuple):
    """Container for ρ and its derivatives.
    
    Attributes:
        rho: ρ(u) = Σ_q p_q(u) σ²(z_q), the expected noise variance
        rho_prime: ρ'(u) = (μ_zs - μ_z μ_s) / τ², first derivative
        rho_double_prime: ρ''(u) = (μ_(z-μ)²s - V_z μ_s) / τ⁴, second derivative
    """
    rho: torch.Tensor
    rho_prime: torch.Tensor
    rho_double_prime: torch.Tensor


def compute_rho_and_derivatives(
    u: torch.Tensor,
    grid: torch.Tensor,
    sigma2_levels: torch.Tensor,
    tau: float,
) -> RhoDerivatives:
    """Compute ρ(u) and its derivatives for the noise-aware objective.
    
    The soft-quantization expected noise variance is:
        ρ(u) = Σ_q p_q(u) σ²(z_q)
    
    where p_q(u) are soft assignment probabilities.
    
    Derivatives are computed using the following moments:
        μ_z = Σ_q p_q z_q                    (mean of z under p)
        V_z = Σ_q p_q (z_q - μ_z)²           (variance of z under p)
        μ_s = Σ_q p_q s_q                    (mean of σ², this is ρ)
        μ_zs = Σ_q p_q z_q s_q               (correlation of z and σ²)
        μ_(z-μ)²s = Σ_q p_q (z_q - μ_z)² s_q (weighted second moment)
    
    Then:
        ρ = μ_s
        ρ' = (μ_zs - μ_z μ_s) / τ²
        ρ'' = (μ_(z-μ)²s - V_z μ_s) / τ⁴
    
    Args:
        u: Input values, shape [...]
        grid: Quantization grid z_q, shape [G]
        sigma2_levels: Noise variance σ²(z_q) for each grid point, shape [G]
        tau: Temperature parameter
        
    Returns:
        RhoDerivatives named tuple with rho, rho_prime, rho_double_prime,
        each of shape [...] matching input u.
        
    Note:
        The derivatives are derived from the chain rule on the softmax.
        See the mathematical derivation in the module docstring.
    """
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    
    # Compute soft assignments p_q(u), shape [..., G]
    p = compute_soft_assign(u, grid, tau)
    
    # Broadcast sigma2_levels to match p: [G] -> [1, ..., 1, G]
    s = sigma2_levels.to(u.device).float()  # s = σ²(z_q)
    z = grid.to(u.device).float()           # z = z_q
    
    # Compute moments (sum over last dimension G)
    # μ_z = E[z] = Σ p_q z_q
    mu_z = (p * z).sum(dim=-1)  # [...]
    
    # V_z = Var[z] = E[(z - μ_z)²] = Σ p_q (z_q - μ_z)²
    z_centered = z - mu_z.unsqueeze(-1)  # [..., G]
    V_z = (p * z_centered ** 2).sum(dim=-1)  # [...]
    
    # μ_s = E[s] = Σ p_q s_q = ρ
    mu_s = (p * s).sum(dim=-1)  # [...]
    
    # μ_zs = E[z·s] = Σ p_q z_q s_q
    mu_zs = (p * z * s).sum(dim=-1)  # [...]
    
    # μ_(z-μ)²s = E[(z - μ_z)² · s] = Σ p_q (z_q - μ_z)² s_q
    mu_z_centered_sq_s = (p * z_centered ** 2 * s).sum(dim=-1)  # [...]
    
    # Compute ρ and derivatives
    rho = mu_s
    
    # ρ' = (μ_zs - μ_z · μ_s) / τ²
    # This is the covariance of z and s divided by τ²
    tau_sq = tau ** 2
    rho_prime = (mu_zs - mu_z * mu_s) / tau_sq
    
    # ρ'' = (μ_(z-μ)²s - V_z · μ_s) / τ⁴
    tau_fourth = tau ** 4
    rho_double_prime = (mu_z_centered_sq_s - V_z * mu_s) / tau_fourth
    
    return RhoDerivatives(rho=rho, rho_prime=rho_prime, rho_double_prime=rho_double_prime)


def noise_aware_round(
    u: torch.Tensor,
    grid: torch.Tensor,
    sigma2_levels: torch.Tensor,
    sensitivity: torch.Tensor,
    a: torch.Tensor,
) -> torch.Tensor:
    """Noise-aware rounding to nearest quantization level.
    
    Instead of standard nearest-neighbor rounding, finds:
        argmin_z [ (z - u)² / sensitivity + a · σ²(z) ]
    
    This balances quantization error against expected noise contribution,
    potentially choosing a sub-optimal rounding if it has lower noise.
    
    Args:
        u: Values to quantize, shape [...]
        grid: Quantization grid, shape [G]
        sigma2_levels: σ²(z_q) for each grid point, shape [G]
        sensitivity: Inverse Hessian diagonal (controls quantization error weight),
                    shape [...] matching u
        a: Hessian diagonal a_i = H_ii (noise sensitivity), shape [...] matching u
        
    Returns:
        Quantized values from grid, shape [...] matching u
    """
    # Expand for broadcasting
    u_expanded = u.unsqueeze(-1)  # [..., 1]
    sens_expanded = sensitivity.unsqueeze(-1)  # [..., 1]
    a_expanded = a.unsqueeze(-1)  # [..., 1]
    
    # Move grid and sigma2 to same device
    z = grid.to(u.device).float()
    s = sigma2_levels.to(u.device).float()
    
    # Compute objective for each grid point
    # (z - u)² / sensitivity + a · σ²(z)
    quant_error = (z - u_expanded) ** 2 / (sens_expanded + 1e-10)
    noise_penalty = a_expanded * s
    
    objective = quant_error + noise_penalty  # [..., G]
    
    # Find argmin
    min_indices = objective.argmin(dim=-1)  # [...]
    
    # Select corresponding grid values
    return z[min_indices]


# =============================================================================
# NA-GPTQ Quantizer
# =============================================================================


class NAGPTQQuantizer(GPTQQuantizer):
    """Noise-Aware GPTQ (NA-GPTQ) quantizer.
    
    Extends GPTQ with noise-aware optimization that accounts for deployment-time
    analog noise. The key modifications are:
    
    1. **Noise-aware rounding**: Instead of nearest rounding, uses
       argmin_z [(z - u)²/sensitivity + a_i σ²(z)]
       
    2. **Noise-aware weight update**: Solves
       (H_FF + 0.5 D) δ = H_FS Δ_S - 0.5 g
       where g_j = a_j ρ'(w_j), D_jj = a_j ρ''(w_j)
    
    The sigma model defines how noise variance depends on quantization level,
    supporting constant, affine, power, and lookup table models.
    
    Attributes:
        tau: Soft assignment temperature (relative to quantization step)
        sigma_model: Model for computing σ²(z) at each level
        use_second_derivative: Whether to include ρ'' in updates
        diag_floor: Minimum value for diagonal D (numerical stability)
    """
    
    def __init__(
        self,
        bits: int = 4,
        group_size: int = 128,
        symmetric: bool = True,  # NA-GPTQ uses symmetric by default
        damp_percent: float = 0.01,
        block_size: int = 128,
        actorder: bool = False,
        # NA-GPTQ specific parameters
        tau: float = 0.1,
        sigma_model: str = "affine",
        sigma_params: Optional[Dict] = None,
        sigma2_table: Optional[torch.Tensor] = None,
        use_second_derivative: bool = False,
        diag_floor: float = 1e-8,
        **kwargs,
    ):
        """Initialize NA-GPTQ quantizer.
        
        Args:
            bits: Quantization bit width (2, 3, 4, or 8)
            group_size: Group size for quantization (-1 for per-channel)
            symmetric: Use symmetric quantization (recommended for NA-GPTQ)
            damp_percent: Dampening percentage for Hessian diagonal
            block_size: Block size for processing columns
            actorder: Use activation-aware ordering
            tau: Soft assignment temperature for ρ computation.
                 Relative to step size; smaller = sharper assignments.
            sigma_model: Type of sigma model ("constant", "affine", "power", "lookup")
            sigma_params: Parameters for sigma model, e.g.:
                         {"sigma0": 0.01, "alpha": 0.05} for affine
            sigma2_table: Lookup table for "lookup" sigma model
            use_second_derivative: Include ρ'' term in update step.
                                  False = simpler, more stable.
                                  True = potentially better but needs tuning.
            diag_floor: Floor for D_jj diagonal (prevents numerical issues)
            **kwargs: Additional configuration
        """
        # Force symmetric for NA-GPTQ (simpler grid structure)
        super().__init__(
            bits=bits,
            group_size=group_size,
            symmetric=True,  # Always symmetric for NA-GPTQ
            damp_percent=damp_percent,
            block_size=block_size,
            actorder=actorder,
            **kwargs,
        )
        
        self.tau = tau
        self.use_second_derivative = use_second_derivative
        self.diag_floor = diag_floor
        
        # Store sigma model config (actual model created per-group with correct scale)
        self._sigma_model_type = sigma_model
        self._sigma_params = sigma_params or {}
        self._sigma2_table = sigma2_table
        
        logger.info(f"Initialized NAGPTQQuantizer")
        logger.info(f"  tau={tau}, sigma_model={sigma_model}")
        logger.info(f"  use_second_derivative={use_second_derivative}, diag_floor={diag_floor}")
    
    def _create_sigma_model(self, scale: float) -> SigmaModel:
        """Create sigma model with the given scale.
        
        Args:
            scale: Quantization scale (step size Δ)
            
        Returns:
            Configured SigmaModel instance
        """
        return create_sigma_model(
            model_type=self._sigma_model_type,
            params=self._sigma_params,
            sigma2_table=self._sigma2_table,
            bits=self.bits,
            scale=scale,
        )
    
    def _quantize_weight(
        self,
        W: torch.Tensor,
        H: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Quantize a weight matrix using noise-aware GPTQ algorithm.
        
        This is the core NA-GPTQ algorithm that replaces the standard GPTQ
        quantization loop with noise-aware rounding and update steps.
        
        Algorithm overview:
        1. For each column i (in order, possibly activation-ordered):
           a. Compute scale for current group
           b. Noise-aware rounding: argmin_z [(z - w_i)² / Hinv[i,i] + a_i σ²(z)]
           c. Compute quantization error Δ_i = w_i - ŵ_i
           d. Noise-aware update for remaining columns:
              (H_FF + 0.5 D) δ = H_FS Δ_S - 0.5 g
              where g_j = a_j ρ'(w_j), D_jj = a_j ρ''(w_j) if use_second_derivative else 0
              
        Args:
            W: Weight matrix [out_features, in_features]
            H: Hessian matrix [in_features, in_features]
            
        Returns:
            Tuple of (quantized_weights, scales, zeros)
            zeros is None for symmetric quantization.
        """
        device = W.device
        out_features, in_features = W.shape
        
        # Compute Hessian inverse for GPTQ-style updates
        try:
            Hinv = torch.linalg.inv(H)
        except RuntimeError:
            # Fallback: add more dampening
            logger.warning("Hessian inversion failed, adding extra dampening")
            H = H + 0.1 * torch.eye(H.shape[0], device=device)
            Hinv = torch.linalg.inv(H)
        
        # Get diagonal sensitivities a_i = H_ii (used in noise penalty)
        H_diag = torch.diag(H)  # [in_features]
        # Get inverse Hessian diagonal (used as sensitivity in rounding, same as GPTQ)
        Hinv_diag = torch.diag(Hinv)  # [in_features]
        
        # Determine quantization order
        if self.actorder:
            perm = self._find_optimal_order(H)
            W = W[:, perm]
            H = H[perm][:, perm]
            Hinv = Hinv[perm][:, perm]
            H_diag = H_diag[perm]
            Hinv_diag = Hinv_diag[perm]
        else:
            perm = None
        
        # Prepare output tensors
        q_max = 2 ** (self.bits - 1) - 1
        num_groups = (in_features + self.group_size - 1) // self.group_size
        Q = torch.zeros_like(W)
        scales = torch.zeros(out_features, num_groups, device=device)
        
        # Pre-compute q_values (integer grid indices, same for all groups)
        q_values = torch.arange(-q_max, q_max + 1, device=device, dtype=torch.float32)
        
        # Create sigma model with scale=1.0 (we'll pass actual z values)
        sigma_model = self._create_sigma_model(scale=1.0)
        
        # Process in blocks
        for block_start in range(0, in_features, self.block_size):
            block_end = min(block_start + self.block_size, in_features)
            
            # Get the block weights (will be modified during updates)
            W_block = W[:, block_start:block_end].clone()
            
            for col in range(block_start, block_end):
                col_local = col - block_start
                group_idx = col // self.group_size
                
                # Compute scale for this group if at group start
                if col % self.group_size == 0:
                    group_start = col
                    group_end = min(col + self.group_size, in_features)
                    W_group = W[:, group_start:group_end]
                    
                    # Symmetric quantization: scale = max(|W|) / Qmax (per row)
                    max_val = W_group.abs().max(dim=1, keepdim=True)[0]
                    scale = (max_val / q_max).squeeze()
                    scale = scale.clamp(min=1e-8)
                    scales[:, group_idx] = scale
                
                # Get current weight column
                w = W_block[:, col_local]  # [out_features]
                
                # Get sensitivity for this column: Hinv[col, col] (same as GPTQ uses)
                # This is the natural sensitivity from the GPTQ algorithm
                sensitivity = Hinv_diag[col]  # scalar tensor
                a_col = H_diag[col]  # scalar tensor (noise sensitivity)
                
                # Noise-aware rounding
                # argmin_z [(z - w)² / sensitivity + a_i σ²(z)]
                scale_col = scales[:, group_idx]  # [out_features]
                
                # Compute sigma² for actual z values (z = scale_col * q)
                # Build per-row z values: [out_features, num_levels]
                z_grid = scale_col.unsqueeze(-1) * q_values.unsqueeze(0)  # [out, num_levels]
                sigma2_grid = sigma_model.sigma_squared(z_grid)  # [out, num_levels]
                
                # Find noise-aware optimal quantization
                q = self._noise_aware_round_perrow(
                    w, scale_col, q_values, sigma2_grid, sensitivity, a_col, q_max
                )
                Q[:, col] = q
                
                # Dequantize for error computation
                w_hat_dequant = q * scale_col
                
                # Compute quantization error Δ = w - ŵ
                err = w - w_hat_dequant  # [out_features]
                
                # === Noise-aware update for remaining weights in block ===
                if col_local < block_end - block_start - 1:
                    hinv_col_col = Hinv[col, col]
                    
                    if hinv_col_col > 1e-10:
                        hinv_col_rest = Hinv[col, col + 1:block_end]  # [remaining_cols]
                        
                        # Compute noise-aware update:
                        # (H_FF + 0.5 D) δ = H_FS Δ - 0.5 g
                        # g_j = a_j ρ'(w_j), D_jj = a_j ρ''(w_j)
                        
                        # Get remaining weights
                        W_rest = W_block[:, col_local + 1:]  # [out, remaining]
                        remaining_cols = W_rest.shape[1]
                        
                        # Get a_j for remaining columns
                        a_rest = H_diag[col + 1:block_end]  # [remaining_cols]
                        H_rest_diag = H_diag[col + 1:block_end]  # [remaining_cols]
                        
                        # Compute ρ' (and optionally ρ'') for remaining weights
                        # Use per-column scale (from the appropriate group) for tau scaling
                        # For simplicity, use mean scale for tau
                        mean_scale = scale_col.mean()
                        tau_eff = self.tau * mean_scale  # τ relative to step size
                        
                        # Build grid for rho computation (use mean scale)
                        grid_for_rho = mean_scale * q_values
                        sigma2_for_rho = sigma_model.sigma_squared(grid_for_rho)
                        
                        # Compute rho derivatives for all remaining weights
                        W_rest_flat = W_rest.flatten()
                        rho_derivs = compute_rho_and_derivatives(
                            W_rest_flat, grid_for_rho, sigma2_for_rho, tau_eff
                        )
                        
                        # Reshape to [out, remaining]
                        rho_prime_mat = rho_derivs.rho_prime.view(out_features, remaining_cols)
                        
                        # g_j = a_j ρ'(w_j) - average across output channels
                        g = (rho_prime_mat * a_rest.unsqueeze(0)).mean(dim=0)  # [remaining]
                        
                        if self.use_second_derivative:
                            rho_dbl_prime_mat = rho_derivs.rho_double_prime.view(out_features, remaining_cols)
                            D_diag = (rho_dbl_prime_mat * a_rest.unsqueeze(0)).mean(dim=0)  # [remaining]
                            D_diag = D_diag.clamp(min=self.diag_floor)
                        else:
                            D_diag = torch.zeros(remaining_cols, device=device)
                        
                        # Solve (H_FF + 0.5 D) δ = H_FS Δ - 0.5 g
                        # Using diagonal approximation for efficiency:
                        # (H_jj + 0.5 D_jj) δ_j ≈ (Hinv[col,j]/Hinv[col,col]) * err - 0.5 g_j
                        # But H_FF is not diagonal... use approximate approach
                        
                        # Standard GPTQ update coefficient
                        standard_coef = hinv_col_rest / hinv_col_col  # [remaining]
                        
                        # Noise correction: -0.5 g / (H_jj + 0.5 D_jj)
                        # Approximate with diagonal: δ_noise ≈ -0.5 g / (H_jj + 0.5 D_jj)
                        denom = H_rest_diag + 0.5 * D_diag + 1e-10
                        
                        # Scale factor for the standard update when D > 0
                        # (H_jj + 0.5 D_jj)^{-1} / H_jj^{-1} = H_jj / (H_jj + 0.5 D_jj)
                        scale_factor = H_rest_diag / denom
                        
                        # Noise gradient correction
                        noise_grad_correction = 0.5 * g / denom
                        
                        # Combined update: scale_factor * standard_update - noise_grad_correction
                        correction = err.unsqueeze(1) * (
                            scale_factor * standard_coef
                        ).unsqueeze(0) - noise_grad_correction.unsqueeze(0)
                        
                        W_block[:, col_local + 1:] -= correction
        
        # Reorder back if needed
        if perm is not None:
            inv_perm = torch.argsort(perm)
            Q = Q[:, inv_perm]
        
        # Return None for zeros (symmetric quantization)
        return Q, scales, None
    
    def _noise_aware_round_perrow(
        self,
        w: torch.Tensor,
        scale: torch.Tensor,
        q_values: torch.Tensor,
        sigma2_grid: torch.Tensor,
        sensitivity: torch.Tensor,
        a: torch.Tensor,
        q_max: int,
    ) -> torch.Tensor:
        """Noise-aware rounding with per-row scales and sigma² values.
        
        Finds argmin_q [(scale*q - w)² / sensitivity + a · σ²(scale*q)]
        
        Optimization: Instead of checking all levels, we only check the 
        3 nearest levels (floor, round, ceil). The noise penalty is typically
        monotonic in |z|, so the optimal choice is usually within ±1 of nearest.
        
        Args:
            w: Weight values in original units, shape [out_features]
            scale: Per-row scale factors, shape [out_features]
            q_values: Integer quantization levels, shape [num_levels]
            sigma2_grid: σ² for each (row, level) pair, shape [out_features, num_levels]
            sensitivity: Hinv[col,col], scalar tensor (same as GPTQ uses)
            a: H_ii for the column, scalar tensor (noise sensitivity)
            q_max: Maximum quantization level
            
        Returns:
            Optimal quantization levels (integer q values), shape [out_features]
        """
        num_levels = len(q_values)
        
        # Normalize w to q-space for each row
        w_normalized = w / scale  # [out_features]
        
        # Get the three candidate levels: floor, round, ceil
        q_floor = torch.floor(w_normalized).clamp(-q_max, q_max)
        q_round = torch.round(w_normalized).clamp(-q_max, q_max)
        q_ceil = torch.ceil(w_normalized).clamp(-q_max, q_max)
        
        # Stack candidates: [out_features, 3]
        q_candidates = torch.stack([q_floor, q_round, q_ceil], dim=-1)
        
        # Get sigma² for each candidate (per-row)
        # Map q values to indices: idx = q + q_max (since q_values starts at -q_max)
        indices = (q_candidates + q_max).long().clamp(0, num_levels - 1)  # [out_features, 3]
        
        # Gather sigma² values: sigma2_grid[i, indices[i, j]]
        sigma2_candidates = sigma2_grid.gather(1, indices)  # [out_features, 3]
        
        # Compute z values for candidates: z = scale * q
        z_candidates = scale.unsqueeze(-1) * q_candidates  # [out_features, 3]
        
        # Compute objective: (z - w)² / sensitivity + a · σ²(z)
        quant_error = (z_candidates - w.unsqueeze(-1)) ** 2 / (sensitivity + 1e-10)
        noise_penalty = a * sigma2_candidates
        
        objective = quant_error + noise_penalty  # [out_features, 3]
        
        # Find best among 3 candidates
        best_idx = objective.argmin(dim=-1)  # [out_features]
        
        # Gather the best q value for each row
        return q_candidates.gather(1, best_idx.unsqueeze(-1)).squeeze(-1)
    
    def _quantize_linear_layer(
        self,
        parent_module: nn.Module,
        name: str,
        linear: nn.Linear,
        layer_inputs: List[torch.Tensor],
        device: torch.device,
    ) -> None:
        """Quantize a single linear layer using NA-GPTQ.
        
        This method overrides the parent to use noise-aware quantization.
        The Hessian computation and layer replacement are inherited from GPTQ.
        
        Args:
            parent_module: Parent module containing the linear layer
            name: Name of the linear layer
            linear: The Linear layer to quantize
            layer_inputs: Captured inputs for Hessian computation
            device: Device for computation
        """
        layer_start = time.perf_counter()
        
        if _DEBUG:
            logger.info(f"  [DEBUG] NA-GPTQ quantizing {name}: "
                       f"{linear.in_features} -> {linear.out_features}, "
                       f"{len(layer_inputs)} calibration inputs")
        else:
            logger.debug(f"  NA-GPTQ quantizing {name}: {linear.in_features} -> {linear.out_features}")
        
        # Get the weight
        W = linear.weight.data.clone().float()
        
        # Compute Hessian approximation (same as GPTQ)
        with Timer("Hessian computation", enabled=_DEBUG):
            H = self._compute_hessian(layer_inputs, linear, device)
        
        # Quantize weight using NA-GPTQ algorithm
        with Timer("Weight quantization", enabled=_DEBUG):
            Q, scales, zeros = self._quantize_weight(W, H)
        
        # Create quantized layer
        qlayer = QuantizedLinear(
            linear.in_features,
            linear.out_features,
            bits=self.bits,
            group_size=self.group_size,
            bias=linear.bias is not None,
        )
        
        # Copy quantized weights (Q contains signed integers [-q_max, q_max])
        qlayer.qweight.copy_(Q.view(qlayer.qweight.shape).to(torch.int8))
        qlayer.scales.copy_(scales.view(qlayer.scales.shape).to(torch.float16))
        
        # For symmetric quantization with signed qweights:
        # dequantize as: weight = qweight * scale (zeros = 0)
        # QuantizedLinear uses: weight = (qweight - zeros) * scale
        # So we set zeros = 0 for correct symmetric dequantization
        qlayer.zeros.fill_(0)
        
        if linear.bias is not None:
            qlayer.bias.copy_(linear.bias.data)
        
        # Replace the linear layer
        self._replace_layer(parent_module, name, qlayer.to(device))
        
        if _DEBUG:
            elapsed = time.perf_counter() - layer_start
            logger.info(f"  [DEBUG] {name} total: {elapsed:.3f}s")

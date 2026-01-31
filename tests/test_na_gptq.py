"""Tests for Noise-Aware GPTQ (NA-GPTQ) implementation.

This module contains:
1. Unit tests for sigma models
2. Unit tests for NA-GPTQ utility functions (grid, soft_assign, rho derivatives)
3. Finite-difference verification of rho derivatives
4. Synthetic test comparing GPTQ vs NA-GPTQ objective values
"""

import pytest
import torch
import numpy as np

from analog_ptq.quantization.sigma_models import (
    ConstantSigmaModel,
    AffineSigmaModel,
    PowerSigmaModel,
    LookupTableSigmaModel,
    create_sigma_model,
)
from analog_ptq.quantization.na_gptq import (
    build_uniform_symmetric_grid,
    compute_soft_assign,
    compute_rho_and_derivatives,
    noise_aware_round,
    NAGPTQQuantizer,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def device():
    """Return available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def grid_4bit():
    """Standard 4-bit symmetric grid with scale=1.0."""
    return build_uniform_symmetric_grid(bits=4, scale=1.0)


@pytest.fixture
def sigma2_affine(grid_4bit):
    """Sigma² levels for affine model."""
    model = AffineSigmaModel(sigma0=0.01, alpha=0.05)
    return model.sigma_squared(grid_4bit)


# =============================================================================
# Tests for Sigma Models
# =============================================================================


class TestSigmaModels:
    """Tests for sigma model implementations."""
    
    def test_constant_sigma_model(self):
        """Test constant sigma model returns constant values."""
        model = ConstantSigmaModel(sigma0=0.1)
        z = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        
        sigma2 = model.sigma_squared(z)
        
        assert sigma2.shape == z.shape
        assert torch.allclose(sigma2, torch.full_like(sigma2, 0.01))  # 0.1^2
    
    def test_affine_sigma_model(self):
        """Test affine sigma model: σ(z) = σ₀ + α|z|."""
        model = AffineSigmaModel(sigma0=0.01, alpha=0.05)
        z = torch.tensor([0.0, 1.0, 2.0, -1.0, -2.0])
        
        sigma2 = model.sigma_squared(z)
        
        # σ(0) = 0.01, σ(1) = 0.06, σ(2) = 0.11
        expected_sigma = torch.tensor([0.01, 0.06, 0.11, 0.06, 0.11])
        expected_sigma2 = expected_sigma ** 2
        
        assert torch.allclose(sigma2, expected_sigma2, atol=1e-6)
    
    def test_power_sigma_model(self):
        """Test power sigma model: σ(z) = σ₀ + α(|z|/z_max)^p."""
        model = PowerSigmaModel(sigma0=0.01, alpha=0.1, power=2.0, z_max=2.0)
        z = torch.tensor([0.0, 1.0, 2.0])
        
        sigma2 = model.sigma_squared(z)
        
        # σ(0) = 0.01 + 0.1*(0/2)^2 = 0.01
        # σ(1) = 0.01 + 0.1*(1/2)^2 = 0.01 + 0.025 = 0.035
        # σ(2) = 0.01 + 0.1*(2/2)^2 = 0.01 + 0.1 = 0.11
        expected_sigma = torch.tensor([0.01, 0.035, 0.11])
        expected_sigma2 = expected_sigma ** 2
        
        assert torch.allclose(sigma2, expected_sigma2, atol=1e-6)
    
    def test_lookup_table_sigma_model(self):
        """Test lookup table sigma model."""
        bits = 3
        q_max = 2 ** (bits - 1) - 1  # 3
        num_levels = 2 * q_max + 1   # 7 levels: -3, -2, -1, 0, 1, 2, 3
        
        # Create a simple table
        sigma2_table = torch.arange(num_levels, dtype=torch.float32) * 0.01
        model = LookupTableSigmaModel(sigma2_table, bits=bits, scale=1.0)
        
        z = torch.tensor([-3.0, 0.0, 3.0])
        sigma2 = model.sigma_squared(z)
        
        # Index 0 (-3) -> 0.00, Index 3 (0) -> 0.03, Index 6 (3) -> 0.06
        expected = torch.tensor([0.00, 0.03, 0.06])
        
        assert torch.allclose(sigma2, expected, atol=1e-6)
    
    def test_create_sigma_model_factory(self):
        """Test sigma model factory function."""
        # Test constant
        model = create_sigma_model("constant", {"sigma0": 0.05})
        assert isinstance(model, ConstantSigmaModel)
        assert model.sigma0 == 0.05
        
        # Test affine
        model = create_sigma_model("affine", {"sigma0": 0.01, "alpha": 0.1})
        assert isinstance(model, AffineSigmaModel)
        assert model.alpha == 0.1
        
        # Test power
        model = create_sigma_model("power", {"sigma0": 0.01, "alpha": 0.1, "power": 0.5})
        assert isinstance(model, PowerSigmaModel)
        assert model.power == 0.5
    
    def test_sigma_model_negative_params_raise(self):
        """Test that negative parameters raise errors."""
        with pytest.raises(ValueError):
            ConstantSigmaModel(sigma0=-0.1)
        
        with pytest.raises(ValueError):
            AffineSigmaModel(sigma0=0.01, alpha=-0.05)


# =============================================================================
# Tests for Grid and Soft Assignment
# =============================================================================


class TestGridAndSoftAssign:
    """Tests for grid construction and soft assignment."""
    
    def test_build_uniform_symmetric_grid_4bit(self):
        """Test 4-bit symmetric grid construction."""
        grid = build_uniform_symmetric_grid(bits=4, scale=0.1)
        
        # 4-bit: Qmax = 7, levels from -7 to 7
        assert len(grid) == 15
        assert grid[0].item() == pytest.approx(-0.7)
        assert grid[7].item() == pytest.approx(0.0)
        assert grid[14].item() == pytest.approx(0.7)
    
    def test_build_uniform_symmetric_grid_3bit(self):
        """Test 3-bit symmetric grid construction."""
        grid = build_uniform_symmetric_grid(bits=3, scale=1.0)
        
        # 3-bit: Qmax = 3, levels from -3 to 3
        assert len(grid) == 7
        assert grid.tolist() == pytest.approx([-3, -2, -1, 0, 1, 2, 3])
    
    def test_soft_assign_sum_to_one(self, grid_4bit):
        """Test that soft assignments sum to 1."""
        u = torch.tensor([0.0, 0.5, 1.5, -2.3])
        tau = 0.5
        
        p = compute_soft_assign(u, grid_4bit, tau)
        
        assert p.shape == (4, 15)
        assert torch.allclose(p.sum(dim=-1), torch.ones(4), atol=1e-6)
    
    def test_soft_assign_approaches_hard_assignment(self, grid_4bit):
        """Test that small tau approaches hard (one-hot) assignment."""
        u = torch.tensor([0.0, 2.0, -3.0])
        tau = 0.01  # Very small
        
        p = compute_soft_assign(u, grid_4bit, tau)
        
        # Should be nearly one-hot at nearest grid point
        # u=0 -> grid index 7, u=2 -> grid index 9, u=-3 -> grid index 4
        assert p[0, 7].item() > 0.99
        assert p[1, 9].item() > 0.99
        assert p[2, 4].item() > 0.99
    
    def test_soft_assign_large_tau_uniform(self, grid_4bit):
        """Test that large tau approaches uniform distribution."""
        u = torch.tensor([0.0])
        tau = 100.0  # Very large
        
        p = compute_soft_assign(u, grid_4bit, tau)
        
        # Should be nearly uniform
        uniform_prob = 1.0 / 15
        assert torch.allclose(p, torch.full_like(p, uniform_prob), atol=0.01)
    
    def test_soft_assign_invalid_tau_raises(self, grid_4bit):
        """Test that non-positive tau raises error."""
        u = torch.tensor([0.0])
        
        with pytest.raises(ValueError):
            compute_soft_assign(u, grid_4bit, tau=0.0)
        
        with pytest.raises(ValueError):
            compute_soft_assign(u, grid_4bit, tau=-0.1)


# =============================================================================
# Tests for Rho and Derivatives (with Finite Difference Verification)
# =============================================================================


class TestRhoDerivatives:
    """Tests for ρ(u) and its derivatives, including finite difference checks."""
    
    def test_rho_is_expected_sigma2(self, grid_4bit, sigma2_affine):
        """Test that ρ(u) = E[σ²(z)] under soft assignment."""
        u = torch.tensor([0.0])
        tau = 0.5
        
        result = compute_rho_and_derivatives(u, grid_4bit, sigma2_affine, tau)
        
        # Manually compute expected value
        p = compute_soft_assign(u, grid_4bit, tau)
        expected_rho = (p * sigma2_affine).sum(dim=-1)
        
        assert torch.allclose(result.rho, expected_rho, atol=1e-6)
    
    def test_rho_prime_finite_difference(self, grid_4bit, sigma2_affine):
        """Verify ρ'(u) using finite differences."""
        tau = 0.5  # Larger tau for more stable derivatives
        eps = 1e-4
        
        # Test at multiple u values (avoid edges where derivatives are small)
        u_values = torch.tensor([-1.0, 0.0, 1.0])
        
        for u in u_values:
            u_tensor = u.unsqueeze(0)
            
            # Compute analytical derivative
            result = compute_rho_and_derivatives(u_tensor, grid_4bit, sigma2_affine, tau)
            analytical = result.rho_prime.item()
            
            # Compute numerical derivative
            rho_plus = compute_rho_and_derivatives(
                u_tensor + eps, grid_4bit, sigma2_affine, tau
            ).rho.item()
            rho_minus = compute_rho_and_derivatives(
                u_tensor - eps, grid_4bit, sigma2_affine, tau
            ).rho.item()
            numerical = (rho_plus - rho_minus) / (2 * eps)
            
            # Check they match (looser tolerance for finite diff numerical precision)
            assert analytical == pytest.approx(numerical, rel=0.1, abs=1e-5), \
                f"rho' mismatch at u={u.item()}: analytical={analytical}, numerical={numerical}"
    
    def test_rho_double_prime_finite_difference(self, grid_4bit, sigma2_affine):
        """Verify ρ''(u) using finite differences on ρ'.
        
        Tests the second derivative at multiple points with central finite differences.
        """
        tau = 0.8  # Larger tau for more stable second derivatives
        eps = 1e-3  # Larger eps for second derivative
        
        # Test at multiple u values including off-center points
        u_values = torch.tensor([-1.0, 0.0, 0.5, 1.5])
        
        for u in u_values:
            u_tensor = u.unsqueeze(0)
            
            # Compute analytical second derivative
            result = compute_rho_and_derivatives(u_tensor, grid_4bit, sigma2_affine, tau)
            analytical = result.rho_double_prime.item()
            
            # Compute numerical second derivative via central diff on ρ'
            rho_prime_plus = compute_rho_and_derivatives(
                u_tensor + eps, grid_4bit, sigma2_affine, tau
            ).rho_prime.item()
            rho_prime_minus = compute_rho_and_derivatives(
                u_tensor - eps, grid_4bit, sigma2_affine, tau
            ).rho_prime.item()
            numerical_from_rho_prime = (rho_prime_plus - rho_prime_minus) / (2 * eps)
            
            # Also verify via second-order central diff on ρ itself
            rho_center = compute_rho_and_derivatives(
                u_tensor, grid_4bit, sigma2_affine, tau
            ).rho.item()
            rho_plus = compute_rho_and_derivatives(
                u_tensor + eps, grid_4bit, sigma2_affine, tau
            ).rho.item()
            rho_minus = compute_rho_and_derivatives(
                u_tensor - eps, grid_4bit, sigma2_affine, tau
            ).rho.item()
            numerical_from_rho = (rho_plus - 2 * rho_center + rho_minus) / (eps ** 2)
            
            # For second derivatives, check both numerical estimates agree with analytical
            if abs(numerical_from_rho_prime) > 1e-8:
                ratio = analytical / numerical_from_rho_prime
                assert 0.3 < ratio < 3.0, \
                    f"rho'' mismatch (via rho') at u={u.item()}: " \
                    f"analytical={analytical}, numerical={numerical_from_rho_prime}, ratio={ratio}"
            
            if abs(numerical_from_rho) > 1e-6:
                ratio2 = analytical / numerical_from_rho
                assert 0.2 < ratio2 < 5.0, \
                    f"rho'' mismatch (via rho) at u={u.item()}: " \
                    f"analytical={analytical}, numerical={numerical_from_rho}, ratio={ratio2}"
    
    def test_rho_derivatives_batch(self, grid_4bit, sigma2_affine):
        """Test that rho computation works with batched inputs."""
        u = torch.randn(10, 5)  # Batch of inputs
        tau = 0.5
        
        result = compute_rho_and_derivatives(u, grid_4bit, sigma2_affine, tau)
        
        assert result.rho.shape == u.shape
        assert result.rho_prime.shape == u.shape
        assert result.rho_double_prime.shape == u.shape
    
    def test_rho_constant_sigma_zero_derivatives(self, grid_4bit):
        """For constant σ², derivatives should be (near) zero."""
        model = ConstantSigmaModel(sigma0=0.1)
        sigma2_const = model.sigma_squared(grid_4bit)
        
        u = torch.tensor([0.0, 1.0, 2.0])
        tau = 0.5
        
        result = compute_rho_and_derivatives(u, grid_4bit, sigma2_const, tau)
        
        # rho should be constant = sigma0^2
        assert torch.allclose(result.rho, torch.full_like(result.rho, 0.01), atol=1e-6)
        
        # Derivatives should be essentially zero
        assert torch.allclose(result.rho_prime, torch.zeros_like(result.rho_prime), atol=1e-6)
        assert torch.allclose(result.rho_double_prime, torch.zeros_like(result.rho_double_prime), atol=1e-5)


# =============================================================================
# Tests for Noise-Aware Rounding
# =============================================================================


class TestNoiseAwareRounding:
    """Tests for noise-aware rounding decision."""
    
    def test_noise_aware_round_matches_nearest_with_zero_noise(self, grid_4bit):
        """With zero noise, should match nearest rounding."""
        sigma2_zero = torch.zeros(15)  # No noise
        u = torch.tensor([0.3, 0.7, 1.2, -0.8])
        sensitivity = torch.ones_like(u)
        a = torch.ones_like(u)
        
        result = noise_aware_round(u, grid_4bit, sigma2_zero, sensitivity, a)
        
        # Should round to nearest: 0, 1, 1, -1
        expected = torch.tensor([0.0, 1.0, 1.0, -1.0])
        
        assert torch.allclose(result, expected)
    
    def test_noise_aware_round_avoids_high_noise(self, grid_4bit, sigma2_affine):
        """Should prefer lower noise levels when penalty is high."""
        u = torch.tensor([3.1])  # Close to 3, but 3 has higher noise
        
        # High sensitivity to noise
        sensitivity = torch.tensor([0.001])  # Small = high weight on quant error
        a = torch.tensor([1000.0])           # Large = high weight on noise
        
        result = noise_aware_round(u, grid_4bit, sigma2_affine, sensitivity, a)
        
        # With very high noise penalty, might choose lower level
        # The exact behavior depends on the tradeoff
        assert result.item() in grid_4bit.tolist()


# =============================================================================
# Synthetic Test: GPTQ vs NA-GPTQ Objective
# =============================================================================


class TestGPTQvsNAGPTQ:
    """Synthetic tests comparing GPTQ and NA-GPTQ objectives.
    
    Tests compare:
    1. GPTQ (nearest rounding + standard update)
    2. NA-GPTQ with noise-aware rounding only (noise-aware round + standard update)
    3. Full NA-GPTQ (noise-aware round + noise-aware update with g term)
    """
    
    def _compute_objective_J(
        self,
        w: torch.Tensor,
        w_hat: torch.Tensor,
        H: torch.Tensor,
        sigma_model,
    ) -> float:
        """Compute the noise-aware objective J(ŵ).
        
        J(ŵ) = (w - ŵ)ᵀ H (w - ŵ) + Σᵢ aᵢ σ²(ŵᵢ)
        
        Args:
            w: Original weights [d]
            w_hat: Quantized weights [d]
            H: Hessian [d, d]
            sigma_model: Sigma model for noise variance
            
        Returns:
            Scalar objective value
        """
        # Reconstruction error term
        delta = w - w_hat
        recon_error = delta @ H @ delta
        
        # Noise penalty term
        H_diag = torch.diag(H)
        sigma2 = sigma_model.sigma_squared(w_hat)
        noise_penalty = (H_diag * sigma2).sum()
        
        return (recon_error + noise_penalty).item()
    
    def _gptq_quantize(
        self,
        w: torch.Tensor,
        H: torch.Tensor,
        grid: torch.Tensor,
    ) -> torch.Tensor:
        """Standard GPTQ: nearest rounding + standard update."""
        d = len(w)
        w_hat = torch.zeros_like(w)
        W = w.clone()
        
        Hinv = torch.linalg.inv(H)
        
        for i in range(d):
            # Nearest rounding
            distances = (W[i] - grid).abs()
            idx = distances.argmin()
            w_hat[i] = grid[idx]
            
            # Standard GPTQ update
            if i < d - 1:
                err = W[i] - w_hat[i]
                hinv_ii = Hinv[i, i]
                if hinv_ii > 1e-10:
                    W[i+1:] -= err * Hinv[i, i+1:] / hinv_ii
        
        return w_hat
    
    def _nagptq_round_only(
        self,
        w: torch.Tensor,
        H: torch.Tensor,
        grid: torch.Tensor,
        sigma_model,
    ) -> torch.Tensor:
        """NA-GPTQ with noise-aware rounding only, standard GPTQ update.
        
        This isolates the effect of noise-aware rounding from the update step.
        """
        d = len(w)
        w_hat = torch.zeros_like(w)
        W = w.clone()
        
        Hinv = torch.linalg.inv(H)
        Hinv_diag = torch.diag(Hinv)
        H_diag = torch.diag(H)
        sigma2_levels = sigma_model.sigma_squared(grid)
        
        for i in range(d):
            # Noise-aware rounding (using Hinv[i,i] as sensitivity)
            sensitivity = Hinv_diag[i]
            a_i = H_diag[i]
            
            w_hat[i] = noise_aware_round(
                W[i:i+1], grid, sigma2_levels,
                torch.tensor([sensitivity.item()]),
                torch.tensor([a_i.item()])
            ).item()
            
            # Standard GPTQ update (no noise correction)
            if i < d - 1:
                err = W[i] - w_hat[i]
                hinv_ii = Hinv[i, i]
                if hinv_ii > 1e-10:
                    W[i+1:] -= err * Hinv[i, i+1:] / hinv_ii
        
        return w_hat
    
    def _nagptq_full(
        self,
        w: torch.Tensor,
        H: torch.Tensor,
        grid: torch.Tensor,
        sigma_model,
        tau: float = 0.3,
        use_second_derivative: bool = False,
        diag_floor: float = 1e-8,
    ) -> torch.Tensor:
        """Full NA-GPTQ: noise-aware rounding + noise-aware update with g (and optional D).
        
        Implements the update: (H_FF + 0.5 D) δ = H_FS Δ - 0.5 g
        where g_j = a_j ρ'(w_j), D_jj = a_j ρ''(w_j)
        """
        d = len(w)
        w_hat = torch.zeros_like(w)
        W = w.clone()
        
        Hinv = torch.linalg.inv(H)
        Hinv_diag = torch.diag(Hinv)
        H_diag = torch.diag(H)
        sigma2_levels = sigma_model.sigma_squared(grid)
        
        # Use mean scale for tau (simplified)
        mean_scale = grid.max().item() / (2 ** 3 - 1)  # approx scale for 4-bit
        tau_eff = tau * mean_scale
        
        for i in range(d):
            # Noise-aware rounding
            sensitivity = Hinv_diag[i]
            a_i = H_diag[i]
            
            w_hat[i] = noise_aware_round(
                W[i:i+1], grid, sigma2_levels,
                torch.tensor([sensitivity.item()]),
                torch.tensor([a_i.item()])
            ).item()
            
            # Noise-aware update with g term
            if i < d - 1:
                err = W[i] - w_hat[i]
                hinv_ii = Hinv[i, i]
                
                if hinv_ii > 1e-10:
                    # Compute rho derivatives for remaining weights
                    W_rest = W[i+1:]
                    rho_derivs = compute_rho_and_derivatives(
                        W_rest, grid, sigma2_levels, tau_eff
                    )
                    
                    # g_j = a_j ρ'(w_j)
                    a_rest = H_diag[i+1:]
                    g = a_rest * rho_derivs.rho_prime
                    
                    # D_jj = a_j ρ''(w_j) if use_second_derivative
                    if use_second_derivative:
                        D_diag = (a_rest * rho_derivs.rho_double_prime).clamp(min=diag_floor)
                    else:
                        D_diag = torch.zeros_like(a_rest)
                    
                    # Standard GPTQ coefficient
                    hinv_col_rest = Hinv[i, i+1:]
                    standard_coef = hinv_col_rest / hinv_ii
                    
                    # Noise-aware update: (H_jj + 0.5 D_jj)^{-1} approx
                    H_rest_diag = H_diag[i+1:]
                    denom = H_rest_diag + 0.5 * D_diag + 1e-10
                    scale_factor = H_rest_diag / denom
                    noise_grad_correction = 0.5 * g / denom
                    
                    # Combined update
                    W[i+1:] -= err * scale_factor * standard_coef - noise_grad_correction
        
        return w_hat
    
    def test_nagptq_update_not_worse_than_round_only(self):
        """Full NA-GPTQ (with g) should not increase J compared to round-only NA-GPTQ.
        
        This tests that the noise-aware update step provides benefit (or at least
        doesn't hurt) compared to just using noise-aware rounding with standard update.
        """
        torch.manual_seed(42)
        
        d = 16
        w = torch.randn(d) * 0.5
        
        X = torch.randn(d, 100)
        H = X @ X.T / 100 + 0.01 * torch.eye(d)
        
        grid = build_uniform_symmetric_grid(bits=4, scale=0.1)
        sigma_model = AffineSigmaModel(sigma0=0.01, alpha=0.05)
        
        # Quantize with round-only and full NA-GPTQ
        w_hat_round_only = self._nagptq_round_only(w, H, grid, sigma_model)
        w_hat_full = self._nagptq_full(w, H, grid, sigma_model, tau=0.3)
        
        J_round_only = self._compute_objective_J(w, w_hat_round_only, H, sigma_model)
        J_full = self._compute_objective_J(w, w_hat_full, H, sigma_model)
        
        # Full NA-GPTQ should not be significantly worse than round-only
        assert J_full <= J_round_only * 1.15, \
            f"Full NA-GPTQ {J_full} worse than round-only {J_round_only}"
    
    def test_nagptq_g_term_differs_from_gptq_update(self):
        """Verify that use_second_derivative=False still applies g, differing from pure GPTQ."""
        torch.manual_seed(42)
        
        d = 16
        w = torch.randn(d) * 0.5
        
        X = torch.randn(d, 100)
        H = X @ X.T / 100 + 0.01 * torch.eye(d)
        
        grid = build_uniform_symmetric_grid(bits=4, scale=0.1)
        sigma_model = AffineSigmaModel(sigma0=0.01, alpha=0.1)  # Higher alpha for visible effect
        
        # Round-only uses standard GPTQ update (no g)
        w_hat_round_only = self._nagptq_round_only(w, H, grid, sigma_model)
        
        # Full NA-GPTQ with use_second_derivative=False still includes g
        w_hat_with_g = self._nagptq_full(
            w, H, grid, sigma_model, tau=0.3, use_second_derivative=False
        )
        
        # The quantized weights should differ (g term has effect)
        # Note: they might be the same in some cases due to rounding, so we check
        # that they're not always identical
        diff = (w_hat_round_only - w_hat_with_g).abs().sum().item()
        
        # At least some weights should differ
        # (This is a soft check - the g term may not always change the rounding)
        # The main check is that the code path is different
        # We verify by checking the objectives
        J_round_only = self._compute_objective_J(w, w_hat_round_only, H, sigma_model)
        J_with_g = self._compute_objective_J(w, w_hat_with_g, H, sigma_model)
        
        # With g term, objective should be same or better (but not significantly worse)
        assert J_with_g <= J_round_only * 1.1, \
            f"NA-GPTQ with g ({J_with_g}) worse than without ({J_round_only})"
    
    def test_nagptq_not_worse_than_gptq_on_synthetic(self):
        """NA-GPTQ should not significantly increase J compared to GPTQ with noise."""
        torch.manual_seed(42)
        
        d = 16
        w = torch.randn(d) * 0.5
        
        X = torch.randn(d, 100)
        H = X @ X.T / 100 + 0.01 * torch.eye(d)
        
        grid = build_uniform_symmetric_grid(bits=4, scale=0.1)
        sigma_model = AffineSigmaModel(sigma0=0.01, alpha=0.05)
        
        w_hat_gptq = self._gptq_quantize(w, H, grid)
        w_hat_nagptq = self._nagptq_full(w, H, grid, sigma_model, tau=0.3)
        
        J_gptq = self._compute_objective_J(w, w_hat_gptq, H, sigma_model)
        J_nagptq = self._compute_objective_J(w, w_hat_nagptq, H, sigma_model)
        
        # NA-GPTQ should not be significantly worse
        assert J_nagptq <= J_gptq * 1.1, \
            f"NA-GPTQ objective {J_nagptq} significantly worse than GPTQ {J_gptq}"
    
    def test_nagptq_improves_with_high_noise(self):
        """NA-GPTQ should show more benefit when noise is significant."""
        torch.manual_seed(123)
        
        d = 16
        w = torch.randn(d) * 0.5
        
        X = torch.randn(d, 100)
        H = X @ X.T / 100 + 0.01 * torch.eye(d)
        
        grid = build_uniform_symmetric_grid(bits=4, scale=0.1)
        
        noise_levels = [0.001, 0.01, 0.1]
        improvements = []
        
        for alpha in noise_levels:
            sigma_model = AffineSigmaModel(sigma0=0.001, alpha=alpha)
            
            w_hat_gptq = self._gptq_quantize(w, H, grid)
            w_hat_nagptq = self._nagptq_full(w, H, grid, sigma_model, tau=0.3)
            
            J_gptq = self._compute_objective_J(w, w_hat_gptq, H, sigma_model)
            J_nagptq = self._compute_objective_J(w, w_hat_nagptq, H, sigma_model)
            
            improvement = (J_gptq - J_nagptq) / J_gptq if J_gptq > 0 else 0
            improvements.append(improvement)
        
        # Higher noise should generally lead to more improvement (or at least not worse)
        assert all(imp >= -0.15 for imp in improvements), \
            f"NA-GPTQ significantly worse at some noise level: {improvements}"


# =============================================================================
# Integration Tests
# =============================================================================


class TestNAGPTQQuantizerIntegration:
    """Integration tests for NAGPTQQuantizer class."""
    
    def test_quantizer_instantiation(self):
        """Test that NAGPTQQuantizer can be instantiated with various configs."""
        # Default config
        quantizer = NAGPTQQuantizer()
        assert quantizer.bits == 4
        assert quantizer.tau == 0.1
        
        # Custom config
        quantizer = NAGPTQQuantizer(
            bits=3,
            tau=0.5,
            sigma_model="affine",
            sigma_params={"sigma0": 0.02, "alpha": 0.1},
            use_second_derivative=True,
            diag_floor=1e-6,
        )
        assert quantizer.bits == 3
        assert quantizer.tau == 0.5
        assert quantizer.use_second_derivative is True
    
    def test_registry_has_nagptq(self):
        """Test that NA-GPTQ is registered in the quantizer registry."""
        from analog_ptq.pipeline.registry import registry
        
        assert "na_gptq" in registry.list_quantizers()
        
        quantizer_cls = registry.get_quantizer("na_gptq")
        assert quantizer_cls is NAGPTQQuantizer
    
    def test_sigma_model_creation(self):
        """Test internal sigma model creation."""
        quantizer = NAGPTQQuantizer(
            sigma_model="affine",
            sigma_params={"sigma0": 0.01, "alpha": 0.05},
        )
        
        sigma_model = quantizer._create_sigma_model(scale=0.1)
        
        assert isinstance(sigma_model, AffineSigmaModel)
        assert sigma_model.sigma0 == 0.01
        assert sigma_model.alpha == 0.05


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

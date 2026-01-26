"""GPTQ quantization implementation.

This module provides an extensible GPTQ implementation that allows easy
modification of the core algorithm components.

References:
    - GPTQ paper: https://arxiv.org/abs/2210.17323
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from analog_ptq.models.wrapper import ModelWrapper
from analog_ptq.quantization.base import BaseQuantizer
from analog_ptq.quantization.utils import QuantizedLinear, find_linear_layers
from analog_ptq.utils.logging import get_logger


logger = get_logger(__name__)


class GPTQQuantizer(BaseQuantizer):
    """GPTQ (Generative Pre-trained Transformer Quantization) implementation.
    
    This implementation is designed for easy modification and extension.
    Key methods that can be overridden for algorithm customization:
    
    - `_compute_hessian`: Compute the Hessian approximation (H = 2 * X^T * X)
    - `_quantize_weight`: Quantize a weight matrix using the GPTQ algorithm
    - `_find_optimal_order`: Find the optimal quantization order
    
    Attributes:
        bits: Quantization bit width
        group_size: Group size for group-wise quantization
        damp_percent: Dampening factor for Hessian diagonal
        block_size: Block size for blocked quantization
        actorder: Whether to use activation order for quantization
    """
    
    def __init__(
        self,
        bits: int = 4,
        group_size: int = 128,
        symmetric: bool = False,
        damp_percent: float = 0.01,
        block_size: int = 128,
        actorder: bool = False,
        **kwargs,
    ):
        """Initialize GPTQ quantizer.
        
        Args:
            bits: Quantization bit width (2, 3, 4, or 8)
            group_size: Group size for quantization (-1 for per-channel)
            symmetric: Use symmetric quantization
            damp_percent: Dampening percentage for Hessian diagonal
            block_size: Block size for processing columns
            actorder: Use activation-aware ordering
            **kwargs: Additional configuration
        """
        super().__init__(bits=bits, group_size=group_size, symmetric=symmetric, **kwargs)
        
        self.damp_percent = damp_percent
        self.block_size = block_size
        self.actorder = actorder
        
        # Storage for layer inputs during calibration
        self._layer_inputs: Dict[int, List[torch.Tensor]] = {}
        self._hooks = []
    
    def prepare(self, model: Union[PreTrainedModel, ModelWrapper]) -> ModelWrapper:
        """Prepare model for GPTQ quantization.
        
        Sets up the model wrapper and prepares for layer-wise quantization.
        
        Args:
            model: Model to prepare
            
        Returns:
            ModelWrapper ready for quantization
        """
        if isinstance(model, ModelWrapper):
            wrapper = model
        else:
            wrapper = ModelWrapper(model)
        
        wrapper.eval()
        
        logger.info(f"Prepared model for GPTQ quantization")
        logger.info(f"  Found {wrapper.num_layers()} layers to quantize")
        
        return wrapper
    
    def quantize(
        self,
        model: ModelWrapper,
        calibration_data: List[torch.Tensor],
    ) -> ModelWrapper:
        """Apply GPTQ quantization to the model.
        
        Performs layer-wise quantization using calibration data to compute
        Hessian approximations for optimal quantization.
        
        Args:
            model: Prepared ModelWrapper
            calibration_data: List of calibration input tensors
            
        Returns:
            Quantized ModelWrapper
        """
        logger.info(f"Starting GPTQ quantization with {len(calibration_data)} samples")
        
        device = next(model.model.parameters()).device
        dtype = next(model.model.parameters()).dtype
        
        # Process layers sequentially
        for layer_idx in range(model.num_layers()):
            logger.info(f"Quantizing layer {layer_idx + 1}/{model.num_layers()}")
            
            # Get all linear layers in this transformer layer
            linear_layers = model.get_linear_layers(layer_idx)
            
            if not linear_layers:
                logger.warning(f"No linear layers found in layer {layer_idx}")
                continue
            
            # Quantize each linear layer - capture inputs per linear layer
            layer_module = model.get_layer(layer_idx)
            for name, linear in linear_layers.items():
                # Capture inputs specifically for this linear layer
                linear_inputs = self._capture_linear_layer_inputs(
                    model, linear, calibration_data
                )
                
                self._quantize_linear_layer(
                    layer_module,
                    name,
                    linear,
                    linear_inputs,
                    device,
                )
            
            # Clear captured inputs to save memory
                del linear_inputs
            
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        logger.info("GPTQ quantization complete")
        
        return model
    
    def _capture_layer_inputs(
        self,
        model: ModelWrapper,
        layer_idx: int,
        calibration_data: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Capture inputs to a specific layer during calibration.
        
        Args:
            model: The model wrapper
            layer_idx: Index of the layer
            calibration_data: Calibration input data
            
        Returns:
            List of captured layer inputs
        """
        captured_inputs = []
        
        def hook_fn(module, inp, out):
            if isinstance(inp, tuple):
                captured_inputs.append(inp[0].detach())
            else:
                captured_inputs.append(inp.detach())
        
        # Register hook on the layer
        layer = model.get_layer(layer_idx)
        handle = layer.register_forward_hook(hook_fn)
        
        # Run calibration data through the model
        with torch.no_grad():
            for data in calibration_data:
                if data.device != next(model.model.parameters()).device:
                    data = data.to(next(model.model.parameters()).device)
                try:
                    model(data)
                except Exception as e:
                    logger.warning(f"Error during calibration forward: {e}")
                    continue
        
        # Remove hook
        handle.remove()
        
        logger.debug(f"Captured {len(captured_inputs)} inputs for layer {layer_idx}")
        
        return captured_inputs
    
    def _capture_linear_layer_inputs(
        self,
        model: ModelWrapper,
        linear: nn.Linear,
        calibration_data: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Capture inputs to a specific linear layer during calibration.
        
        This captures the actual inputs to the linear layer, which is crucial
        for layers like mlp.down_proj that receive different inputs than the
        transformer layer input.
        
        Args:
            model: The model wrapper
            linear: The linear layer to capture inputs for
            calibration_data: Calibration input data
            
        Returns:
            List of captured linear layer inputs
        """
        captured_inputs = []
        
        def hook_fn(module, inp, out):
            if isinstance(inp, tuple):
                captured_inputs.append(inp[0].detach())
            else:
                captured_inputs.append(inp.detach())
        
        # Register hook on the specific linear layer
        handle = linear.register_forward_hook(hook_fn)
        
        # Run calibration data through the model
        with torch.no_grad():
            for data in calibration_data:
                if data.device != next(model.model.parameters()).device:
                    data = data.to(next(model.model.parameters()).device)
                try:
                    model(data)
                except Exception as e:
                    logger.warning(f"Error during calibration forward: {e}")
                    continue
        
        # Remove hook
        handle.remove()
        
        logger.debug(f"Captured {len(captured_inputs)} inputs for linear layer")
        
        return captured_inputs
    
    def _quantize_linear_layer(
        self,
        parent_module: nn.Module,
        name: str,
        linear: nn.Linear,
        layer_inputs: List[torch.Tensor],
        device: torch.device,
    ) -> None:
        """Quantize a single linear layer using GPTQ.
        
        This method can be overridden to customize the quantization process.
        
        Args:
            parent_module: Parent module containing the linear layer
            name: Name of the linear layer
            linear: The Linear layer to quantize
            layer_inputs: Captured inputs for Hessian computation
            device: Device for computation
        """
        logger.debug(f"  Quantizing {name}: {linear.in_features} -> {linear.out_features}")
        
        # Get the weight
        W = linear.weight.data.clone().float()
        
        # Compute Hessian approximation
        H = self._compute_hessian(layer_inputs, linear, device)
        
        # Quantize weight using GPTQ algorithm
        Q, scales, zeros = self._quantize_weight(W, H)
        
        # Create quantized layer
        qlayer = QuantizedLinear(
            linear.in_features,
            linear.out_features,
            bits=self.bits,
            group_size=self.group_size,
            bias=linear.bias is not None,
        )
        
        # Copy quantized weights
        qlayer.qweight.copy_(Q.view(qlayer.qweight.shape).to(torch.int8))
        qlayer.scales.copy_(scales.view(qlayer.scales.shape).to(torch.float16))
        if zeros is not None:
            qlayer.zeros.copy_(zeros.view(qlayer.zeros.shape).to(torch.float16))
        
        if linear.bias is not None:
            qlayer.bias.copy_(linear.bias.data)
        
        # Replace the linear layer
        self._replace_layer(parent_module, name, qlayer.to(device))
    
    def _compute_hessian(
        self,
        layer_inputs: List[torch.Tensor],
        linear: nn.Linear,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute Hessian approximation for GPTQ.
        
        The Hessian is approximated as H = 2 * X^T * X where X contains
        the activation inputs to the layer.
        
        Override this method to customize Hessian computation.
        
        Args:
            layer_inputs: Captured layer inputs
            linear: The linear layer
            device: Device for computation
            
        Returns:
            Hessian matrix [in_features, in_features]
        """
        nsamples = 0
        H = torch.zeros(
            (linear.in_features, linear.in_features),
            device=device,
            dtype=torch.float32,
        )
        
        for inp in layer_inputs:
            inp = inp.to(device).float()
            
            # Reshape to 2D: [batch * seq, features]
            if inp.dim() == 3:
                inp = inp.reshape(-1, inp.shape[-1])
            
            # Only use the first in_features columns
            inp = inp[:, :linear.in_features]
            
            # Accumulate Hessian: H += X^T @ X
            H += inp.T @ inp
            nsamples += inp.shape[0]
        
        # Average over samples
        H /= nsamples
        
        # Add dampening
        damp = self.damp_percent * torch.diag(H).mean()
        H += damp * torch.eye(H.shape[0], device=device)
        
        return H
    
    def _quantize_weight(
        self,
        W: torch.Tensor,
        H: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Quantize a weight matrix using the GPTQ algorithm.
        
        Override this method to customize the core quantization algorithm.
        
        Args:
            W: Weight matrix [out_features, in_features]
            H: Hessian matrix [in_features, in_features]
            
        Returns:
            Tuple of (quantized_weights, scales, zeros)
        """
        device = W.device
        out_features, in_features = W.shape
        
        # Compute inverse Hessian
        try:
            Hinv = torch.linalg.inv(H)
        except RuntimeError:
            # Fallback: add more dampening
            H += 0.1 * torch.eye(H.shape[0], device=device)
            Hinv = torch.linalg.inv(H)
        
        # Determine quantization order
        if self.actorder:
            perm = self._find_optimal_order(H)
            W = W[:, perm]
            Hinv = Hinv[perm][:, perm]
        else:
            perm = None
        
        # Prepare output tensors
        num_groups = (in_features + self.group_size - 1) // self.group_size
        Q = torch.zeros_like(W)
        scales = torch.zeros(out_features, num_groups, device=device)
        zeros = torch.zeros(out_features, num_groups, device=device)
        
        # Process in blocks
        for block_start in range(0, in_features, self.block_size):
            block_end = min(block_start + self.block_size, in_features)
            
            # Get the block
            W_block = W[:, block_start:block_end].clone()
            
            for col in range(block_start, block_end):
                col_local = col - block_start
                group_idx = col // self.group_size
                
                # Compute quantization parameters for this group if at group start
                if col % self.group_size == 0:
                    group_start = col
                    group_end = min(col + self.group_size, in_features)
                    W_group = W[:, group_start:group_end]
                    
                    if self.symmetric:
                        max_val = W_group.abs().max(dim=1, keepdim=True)[0]
                        scales[:, group_idx] = (max_val / (2 ** (self.bits - 1) - 1)).squeeze()
                        scales[:, group_idx] = scales[:, group_idx].clamp(min=1e-8)
                    else:
                        min_val = W_group.min(dim=1, keepdim=True)[0]
                        max_val = W_group.max(dim=1, keepdim=True)[0]
                        scales[:, group_idx] = ((max_val - min_val) / (2 ** self.bits - 1)).squeeze()
                        scales[:, group_idx] = scales[:, group_idx].clamp(min=1e-8)
                        zeros[:, group_idx] = (-min_val / scales[:, group_idx].unsqueeze(1)).squeeze()
                
                # Get weight column
                w = W_block[:, col_local]
                
                # Quantize
                scale = scales[:, group_idx]
                zero = zeros[:, group_idx]
                
                if self.symmetric:
                    q = torch.round(w / scale)
                    q = q.clamp(-(2 ** (self.bits - 1)), 2 ** (self.bits - 1) - 1)
                else:
                    q = torch.round(w / scale + zero)
                    q = q.clamp(0, 2 ** self.bits - 1)
                
                Q[:, col] = q
                
                # Compute quantization error
                if self.symmetric:
                    w_hat = q * scale
                else:
                    w_hat = (q - zero) * scale
                
                err = w - w_hat
                
                # Update remaining weights in block (error correction)
                if col_local < block_end - block_start - 1:
                    hinv_col = Hinv[col, col]
                    if hinv_col > 0:
                        correction = err.unsqueeze(1) * Hinv[col, col + 1:block_end].unsqueeze(0) / hinv_col
                        W_block[:, col_local + 1:] -= correction
        
        # Reorder back if needed
        if perm is not None:
            inv_perm = torch.argsort(perm)
            Q = Q[:, inv_perm]
        
        return Q, scales, zeros if not self.symmetric else None
    
    def _find_optimal_order(self, H: torch.Tensor) -> torch.Tensor:
        """Find optimal quantization order based on Hessian diagonal.
        
        Columns with higher Hessian diagonal values are quantized first,
        as they have more impact on the loss.
        
        Override this method to customize the ordering strategy.
        
        Args:
            H: Hessian matrix
            
        Returns:
            Permutation tensor
        """
        # Sort by diagonal values (descending)
        diag = torch.diag(H)
        perm = torch.argsort(diag, descending=True)
        return perm
    
    def _replace_layer(
        self,
        parent: nn.Module,
        name: str,
        new_layer: nn.Module,
    ) -> None:
        """Replace a nested layer in a module.
        
        Args:
            parent: Parent module
            name: Dotted name of layer (e.g., "self_attn.q_proj")
            new_layer: New layer to insert
        """
        parts = name.split(".")
        module = parent
        
        # Navigate to parent of target
        for part in parts[:-1]:
            module = getattr(module, part)
        
        # Replace the layer
        setattr(module, parts[-1], new_layer)

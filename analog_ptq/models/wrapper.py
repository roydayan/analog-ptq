"""Model wrapper providing hooks for layer-wise quantization access."""

from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from analog_ptq.utils.logging import get_logger


logger = get_logger(__name__)


class ModelWrapper:
    """Wrapper class that provides hooks for layer-wise access and quantization.
    
    This wrapper allows:
    - Iterating over model layers for sequential quantization
    - Capturing layer inputs/outputs for calibration
    - Replacing layers with quantized versions
    
    Attributes:
        model: The wrapped HuggingFace model
        layers: List of transformer layers (decoder blocks)
        layer_names: Names of the layers in the model
    """
    
    # Common layer name patterns for different model architectures
    LAYER_PATTERNS = [
        "model.layers",           # LLaMA, Mistral, etc.
        "transformer.h",          # GPT-2, GPT-J
        "gpt_neox.layers",        # GPT-NeoX, Pythia
        "transformer.blocks",     # Falcon
        "model.decoder.layers",   # OPT
    ]
    
    def __init__(self, model: PreTrainedModel):
        """Initialize the model wrapper.
        
        Args:
            model: HuggingFace PreTrainedModel to wrap
        """
        self.model = model
        self._hooks: Dict[str, List] = {}
        self._captured_inputs: Dict[str, List[torch.Tensor]] = {}
        self._captured_outputs: Dict[str, List[torch.Tensor]] = {}
        
        # Find and store layer references
        self.layers, self.layer_names = self._find_layers()
        logger.info(f"Found {len(self.layers)} transformer layers")
    
    def _find_layers(self) -> Tuple[nn.ModuleList, List[str]]:
        """Find transformer layers in the model.
        
        Returns:
            Tuple of (layers ModuleList, layer name strings)
        """
        for pattern in self.LAYER_PATTERNS:
            parts = pattern.split(".")
            module = self.model
            
            try:
                for part in parts:
                    module = getattr(module, part)
                
                if isinstance(module, nn.ModuleList):
                    names = [f"{pattern}.{i}" for i in range(len(module))]
                    return module, names
            except AttributeError:
                continue
        
        # Fallback: search for ModuleList with many children
        for name, module in self.model.named_modules():
            if isinstance(module, nn.ModuleList) and len(module) > 1:
                # Check if children look like transformer layers
                first_child = module[0]
                if hasattr(first_child, "self_attn") or hasattr(first_child, "attention"):
                    names = [f"{name}.{i}" for i in range(len(module))]
                    logger.info(f"Found layers at: {name}")
                    return module, names
        
        raise ValueError("Could not find transformer layers in model")
    
    def get_layer(self, index: int) -> nn.Module:
        """Get a specific layer by index.
        
        Args:
            index: Layer index
            
        Returns:
            The transformer layer module
        """
        return self.layers[index]
    
    def set_layer(self, index: int, new_layer: nn.Module) -> None:
        """Replace a layer with a new module.
        
        Args:
            index: Layer index to replace
            new_layer: New module to insert
        """
        self.layers[index] = new_layer
    
    def num_layers(self) -> int:
        """Get the number of transformer layers.
        
        Returns:
            Number of layers
        """
        return len(self.layers)
    
    def get_linear_layers(self, layer_index: int) -> Dict[str, nn.Linear]:
        """Get all linear layers within a transformer layer.
        
        Args:
            layer_index: Index of the transformer layer
            
        Returns:
            Dict mapping layer names to Linear modules
        """
        layer = self.get_layer(layer_index)
        linear_layers = {}
        
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                linear_layers[name] = module
        
        return linear_layers
    
    def register_forward_hook(
        self,
        layer_index: int,
        hook_fn: Callable,
        name: Optional[str] = None,
    ) -> None:
        """Register a forward hook on a layer.
        
        Args:
            layer_index: Index of the layer
            hook_fn: Hook function with signature (module, input, output) -> None
            name: Optional name for the hook (for later removal)
        """
        layer = self.get_layer(layer_index)
        handle = layer.register_forward_hook(hook_fn)
        
        key = name or f"layer_{layer_index}"
        if key not in self._hooks:
            self._hooks[key] = []
        self._hooks[key].append(handle)
    
    def remove_hooks(self, name: Optional[str] = None) -> None:
        """Remove registered hooks.
        
        Args:
            name: If provided, remove only hooks with this name.
                  If None, remove all hooks.
        """
        if name is not None:
            handles = self._hooks.pop(name, [])
            for handle in handles:
                handle.remove()
        else:
            for handles in self._hooks.values():
                for handle in handles:
                    handle.remove()
            self._hooks.clear()
    
    def capture_layer_io(
        self,
        layer_index: int,
        capture_input: bool = True,
        capture_output: bool = True,
    ) -> None:
        """Set up hooks to capture layer inputs and outputs.
        
        Args:
            layer_index: Index of the layer to capture
            capture_input: Whether to capture inputs
            capture_output: Whether to capture outputs
        """
        key = f"capture_{layer_index}"
        self._captured_inputs[key] = []
        self._captured_outputs[key] = []
        
        def hook_fn(module, inp, out):
            if capture_input:
                # inp is a tuple, typically (hidden_states, attention_mask, ...)
                if isinstance(inp, tuple) and len(inp) > 0:
                    self._captured_inputs[key].append(inp[0].detach().cpu())
            if capture_output:
                # out can be a tuple or tensor
                if isinstance(out, tuple):
                    self._captured_outputs[key].append(out[0].detach().cpu())
                else:
                    self._captured_outputs[key].append(out.detach().cpu())
        
        self.register_forward_hook(layer_index, hook_fn, name=key)
    
    def get_captured_io(
        self,
        layer_index: int,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Get captured inputs and outputs for a layer.
        
        Args:
            layer_index: Index of the layer
            
        Returns:
            Tuple of (inputs list, outputs list)
        """
        key = f"capture_{layer_index}"
        return (
            self._captured_inputs.get(key, []),
            self._captured_outputs.get(key, []),
        )
    
    def clear_captured_io(self, layer_index: Optional[int] = None) -> None:
        """Clear captured inputs/outputs.
        
        Args:
            layer_index: If provided, clear only for this layer.
                        If None, clear all captured data.
        """
        if layer_index is not None:
            key = f"capture_{layer_index}"
            self._captured_inputs.pop(key, None)
            self._captured_outputs.pop(key, None)
            self.remove_hooks(key)
        else:
            self._captured_inputs.clear()
            self._captured_outputs.clear()
            self.remove_hooks()
    
    def forward(self, *args, **kwargs):
        """Forward pass through the model.
        
        Args:
            *args: Positional arguments for model forward
            **kwargs: Keyword arguments for model forward
            
        Returns:
            Model output
        """
        return self.model(*args, **kwargs)
    
    def generate(self, *args, **kwargs):
        """Generate text using the model.
        
        Args:
            *args: Positional arguments for model.generate
            **kwargs: Keyword arguments for model.generate
            
        Returns:
            Generated token ids
        """
        return self.model.generate(*args, **kwargs)
    
    def __call__(self, *args, **kwargs):
        """Call the model forward."""
        return self.forward(*args, **kwargs)
    
    def to(self, device: Union[str, torch.device]) -> "ModelWrapper":
        """Move model to device.
        
        Args:
            device: Target device
            
        Returns:
            Self for chaining
        """
        self.model.to(device)
        return self
    
    def eval(self) -> "ModelWrapper":
        """Set model to evaluation mode.
        
        Returns:
            Self for chaining
        """
        self.model.eval()
        return self
    
    def train(self, mode: bool = True) -> "ModelWrapper":
        """Set model training mode.
        
        Args:
            mode: Training mode flag
            
        Returns:
            Self for chaining
        """
        self.model.train(mode)
        return self

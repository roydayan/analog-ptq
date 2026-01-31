"""Base quantizer class defining the interface for all quantization methods."""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import PreTrainedModel

from analog_ptq.models.wrapper import ModelWrapper
from analog_ptq.utils.logging import get_logger


logger = get_logger(__name__)


def check_cached_model(output_path: Union[str, Path]) -> bool:
    """Check if a cached quantized model exists at the given path.
    
    A model is considered cached if:
    1. The directory exists
    2. It contains a quantization_config.json
    3. It contains either model.safetensors or pytorch_model.bin
    
    Args:
        output_path: Directory path to check
        
    Returns:
        True if a cached model exists, False otherwise
    """
    output_path = Path(output_path)
    
    if not output_path.exists():
        return False
    
    config_path = output_path / "quantization_config.json"
    if not config_path.exists():
        return False
    
    # Check for model weights
    safetensors_path = output_path / "model.safetensors"
    pytorch_path = output_path / "pytorch_model.bin"
    
    return safetensors_path.exists() or pytorch_path.exists()


def load_cached_quantized_model(
    model_path: Union[str, Path],
    device_map: Union[str, Dict] = "auto",
    dtype: str = "float16",
) -> Tuple[PreTrainedModel, Dict[str, Any]]:
    """Load a previously quantized model from disk.
    
    This function loads a model that was previously quantized and saved,
    restoring the QuantizedLinear layers.
    
    Args:
        model_path: Path to the quantized model directory
        device_map: Device mapping for model loading
        dtype: Data type for loading
        
    Returns:
        Tuple of (loaded model, quantization config dict)
        
    Raises:
        FileNotFoundError: If model path doesn't exist or is incomplete
    """
    from analog_ptq.models.loader import load_model
    from analog_ptq.quantization.utils import QuantizedLinear
    
    model_path = Path(model_path)
    
    # Load quantization config
    config_path = model_path / "quantization_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No quantization config found at {config_path}")
    
    with open(config_path) as f:
        quant_config = json.load(f)
    
    original_model = quant_config.get("original_model")
    if not original_model:
        raise ValueError("quantization_config.json missing 'original_model' field")
    
    bits = quant_config.get("bits", 4)
    group_size = quant_config.get("group_size", 128)
    
    logger.info(f"Loading cached quantized model from {model_path}")
    logger.info(f"  Original model: {original_model}")
    logger.info(f"  Quantization: {bits}-bit, group_size={group_size}")
    
    # Load base model architecture
    model = load_model(
        original_model,
        dtype=dtype,
        device_map=device_map,
    )
    
    # Find and load weights
    safetensors_path = model_path / "model.safetensors"
    pytorch_path = model_path / "pytorch_model.bin"
    
    if safetensors_path.exists():
        from safetensors.torch import load_file
        state_dict = load_file(str(safetensors_path))
    elif pytorch_path.exists():
        state_dict = torch.load(str(pytorch_path), map_location="cpu")
    else:
        raise FileNotFoundError(f"No model weights found at {model_path}")
    
    # Find quantized layers and restore them
    quantized_layers = set()
    for key in state_dict.keys():
        if ".qweight" in key:
            layer_path = key.rsplit(".qweight", 1)[0]
            quantized_layers.add(layer_path)
    
    for layer_path in quantized_layers:
        parts = layer_path.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        
        layer_name = parts[-1]
        original_layer = getattr(parent, layer_name)
        
        # Create QuantizedLinear and load weights
        qlayer = QuantizedLinear(
            in_features=original_layer.in_features,
            out_features=original_layer.out_features,
            bits=bits,
            group_size=group_size,
            bias=original_layer.bias is not None,
        )
        
        qlayer.qweight.copy_(state_dict[f"{layer_path}.qweight"])
        qlayer.scales.copy_(state_dict[f"{layer_path}.scales"])
        qlayer.zeros.copy_(state_dict[f"{layer_path}.zeros"])
        if original_layer.bias is not None and f"{layer_path}.bias" in state_dict:
            qlayer.bias.copy_(state_dict[f"{layer_path}.bias"])
        
        # Get device from model
        device = next(model.parameters()).device
        setattr(parent, layer_name, qlayer.to(device))
    
    model.eval()
    
    logger.info(f"  Loaded {len(quantized_layers)} quantized layers")
    
    return model, quant_config


class BaseQuantizer(ABC):
    """Abstract base class for quantization methods.
    
    All quantization implementations should inherit from this class and
    implement the required abstract methods.
    
    Attributes:
        bits: Number of bits for quantization
        group_size: Group size for group-wise quantization (-1 for per-channel)
        symmetric: Whether to use symmetric quantization
    """
    
    def __init__(
        self,
        bits: int = 4,
        group_size: int = 128,
        symmetric: bool = False,
        **kwargs,
    ):
        """Initialize the quantizer.
        
        Args:
            bits: Number of bits for quantization (typically 2, 3, 4, or 8)
            group_size: Group size for group-wise quantization.
                       Use -1 for per-channel quantization.
            symmetric: Whether to use symmetric quantization around zero
            **kwargs: Additional quantizer-specific parameters
        """
        self.bits = bits
        self.group_size = group_size
        self.symmetric = symmetric
        self.config = kwargs
        
        logger.info(f"Initialized {self.__class__.__name__}")
        logger.info(f"  bits={bits}, group_size={group_size}, symmetric={symmetric}")
    
    @abstractmethod
    def prepare(self, model: Union[PreTrainedModel, ModelWrapper]) -> ModelWrapper:
        """Prepare the model for quantization.
        
        This method should set up any necessary hooks, data structures, or
        modifications needed before quantization begins.
        
        Args:
            model: The model to prepare (will be wrapped if not already)
            
        Returns:
            ModelWrapper containing the prepared model
        """
        pass
    
    @abstractmethod
    def quantize(
        self,
        model: ModelWrapper,
        calibration_data: List[torch.Tensor],
    ) -> ModelWrapper:
        """Apply quantization to the model.
        
        Args:
            model: The prepared model wrapper
            calibration_data: List of input tensors for calibration
            
        Returns:
            ModelWrapper containing the quantized model
        """
        pass
    
    def save(
        self,
        model: Union[PreTrainedModel, ModelWrapper],
        output_path: Union[str, Path],
        save_quantization_config: bool = True,
        original_model: str = None,
    ) -> None:
        """Save the quantized model.
        
        Args:
            model: The quantized model
            output_path: Directory path to save the model
            save_quantization_config: Whether to save quantization config
            original_model: Name/path of original model (for tokenizer loading)
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get the underlying model if wrapped
        if isinstance(model, ModelWrapper):
            hf_model = model.model
        else:
            hf_model = model
        
        logger.info(f"Saving quantized model to {output_path}")
        hf_model.save_pretrained(output_path)
        
        if save_quantization_config:
            self._save_quantization_config(output_path, original_model)
    
    def _save_quantization_config(self, output_path: Path, original_model: str = None) -> None:
        """Save quantization configuration to a JSON file.
        
        Args:
            output_path: Directory to save config
            original_model: Name/path of the original model (for tokenizer loading)
        """
        import json
        
        config = {
            "quantizer": self.__class__.__name__,
            "bits": self.bits,
            "group_size": self.group_size,
            "symmetric": self.symmetric,
            **self.config,
        }
        
        if original_model:
            config["original_model"] = original_model
        
        config_path = output_path / "quantization_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved quantization config to {config_path}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get the quantizer configuration.
        
        Returns:
            Dictionary containing quantizer configuration
        """
        return {
            "quantizer": self.__class__.__name__,
            "bits": self.bits,
            "group_size": self.group_size,
            "symmetric": self.symmetric,
            **self.config,
        }

"""Base quantizer class defining the interface for all quantization methods."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import PreTrainedModel

from analog_ptq.models.wrapper import ModelWrapper
from analog_ptq.utils.logging import get_logger


logger = get_logger(__name__)


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

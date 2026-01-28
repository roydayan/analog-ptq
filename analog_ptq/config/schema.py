"""Configuration schema and validation using Pydantic."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator


class ExperimentMeta(BaseModel):
    """Experiment metadata configuration."""
    
    name: str = Field(..., description="Experiment name")
    output_dir: str = Field("./outputs", description="Output directory")
    seed: int = Field(42, description="Random seed")
    
    @field_validator("output_dir")
    @classmethod
    def validate_output_dir(cls, v: str) -> str:
        """Ensure output dir is a valid path."""
        return str(Path(v).expanduser())


class ModelConfig(BaseModel):
    """Model loading configuration."""
    
    name_or_path: str = Field(..., description="Model name or path")
    dtype: str = Field("float16", description="Model dtype (float16, bfloat16, float32)")
    device_map: Union[str, Dict[str, Any]] = Field("auto", description="Device mapping")
    trust_remote_code: bool = Field(False, description="Trust remote code")
    revision: Optional[str] = Field(None, description="Model revision")
    
    @field_validator("dtype")
    @classmethod
    def validate_dtype(cls, v: str) -> str:
        """Validate dtype string."""
        valid_dtypes = ["float16", "fp16", "bfloat16", "bf16", "float32", "fp32"]
        if v.lower() not in valid_dtypes:
            raise ValueError(f"dtype must be one of {valid_dtypes}")
        return v.lower()


class CalibrationConfig(BaseModel):
    """Calibration data configuration."""
    
    dataset: str = Field("wikitext", description="Dataset name")
    num_samples: int = Field(128, description="Number of calibration samples")
    seq_length: int = Field(2048, description="Sequence length")
    seed: int = Field(42, description="Random seed for sampling")
    
    @field_validator("num_samples")
    @classmethod
    def validate_num_samples(cls, v: int) -> int:
        """Ensure positive number of samples."""
        if v < 1:
            raise ValueError("num_samples must be positive")
        return v


class QuantizationConfig(BaseModel):
    """Quantization method configuration."""
    
    method: str = Field("gptq", description="Quantization method")
    bits: int = Field(4, description="Quantization bits")
    group_size: int = Field(128, description="Group size (-1 for per-channel)")
    symmetric: bool = Field(False, description="Use symmetric quantization")
    damp_percent: float = Field(0.01, description="Dampening percentage for GPTQ")
    block_size: int = Field(128, description="Block size for GPTQ")
    actorder: bool = Field(False, description="Use activation order for GPTQ")
    calibration: CalibrationConfig = Field(
        default_factory=CalibrationConfig,
        description="Calibration configuration"
    )
    extra: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra quantizer-specific parameters"
    )
    
    @field_validator("bits")
    @classmethod
    def validate_bits(cls, v: int) -> int:
        """Validate bit width."""
        if v not in [2, 3, 4, 8]:
            raise ValueError("bits must be 2, 3, 4, or 8")
        return v


class NoiseConfig(BaseModel):
    """Gaussian noise injection configuration.
    
    Configures noise injection into quantized model weights for analog computing simulation.
    The noise standard deviation is computed as a function of the weight values.
    
    Modes:
        - "static": Noise is applied once after quantization (permanent weight perturbation)
        - "dynamic": Noise is applied at inference time during each forward pass
        - "both": Both static and dynamic noise are applied
    """
    
    enabled: bool = Field(True, description="Whether noise injection is enabled")
    function: str = Field(
        "proportional",
        description="Name of the noise function (e.g., 'constant', 'proportional', 'polynomial')"
    )
    function_params: Dict[str, Any] = Field(
        default_factory=lambda: {"scale": 0.1},
        description="Parameters passed to the noise function"
    )
    mode: str = Field(
        "static",
        description="Noise mode: 'static', 'dynamic', or 'both'"
    )
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    target_layers: Optional[List[str]] = Field(
        None,
        description="Optional list of layer name patterns to target (None = all layers)"
    )
    
    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        """Validate noise mode."""
        valid_modes = ["static", "dynamic", "both"]
        if v.lower() not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}")
        return v.lower()


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""
    
    tasks: List[str] = Field(
        default_factory=lambda: ["hellaswag"],
        description="Evaluation tasks"
    )
    batch_size: int = Field(8, description="Evaluation batch size")
    num_fewshot: Optional[int] = Field(None, description="Number of few-shot examples")
    limit: Optional[int] = Field(None, description="Limit number of samples per task")
    device: Optional[str] = Field(None, description="Device for evaluation")
    
    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        """Ensure positive batch size."""
        if v < 1:
            raise ValueError("batch_size must be positive")
        return v


class ExperimentConfig(BaseModel):
    """Full experiment configuration."""
    
    experiment: ExperimentMeta
    model: ModelConfig
    quantization: Optional[QuantizationConfig] = None
    noise: Optional[NoiseConfig] = None
    evaluation: Optional[EvaluationConfig] = None
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ExperimentConfig":
        """Load configuration from a YAML file.
        
        Args:
            path: Path to YAML configuration file
            
        Returns:
            Parsed ExperimentConfig
        """
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to a YAML file.
        
        Args:
            path: Path to save YAML file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)
    
    def model_dump_yaml(self) -> str:
        """Dump configuration as YAML string.
        
        Returns:
            YAML string representation
        """
        return yaml.dump(self.model_dump(), default_flow_style=False, sort_keys=False)


def load_config(path: Union[str, Path]) -> ExperimentConfig:
    """Load experiment configuration from a YAML file.
    
    Args:
        path: Path to YAML configuration file
        
    Returns:
        Parsed ExperimentConfig
        
    Example:
        >>> config = load_config("configs/my_experiment.yaml")
        >>> print(config.model.name_or_path)
    """
    return ExperimentConfig.from_yaml(path)

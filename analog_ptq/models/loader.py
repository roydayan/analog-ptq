"""HuggingFace model loading utilities."""

from typing import Optional, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from analog_ptq.utils.logging import get_logger


logger = get_logger(__name__)


def load_model(
    name_or_path: str,
    dtype: Optional[str] = "float16",
    device_map: Optional[Union[str, dict]] = "auto",
    trust_remote_code: bool = False,
    torch_dtype: Optional[torch.dtype] = None,
    **kwargs,
) -> PreTrainedModel:
    """Load a HuggingFace model for causal language modeling.
    
    Args:
        name_or_path: Model name on HuggingFace Hub or local path
        dtype: Data type string ("float16", "bfloat16", "float32")
        device_map: Device placement strategy ("auto", "cuda", "cpu", or custom dict)
        trust_remote_code: Whether to trust remote code in model repos
        torch_dtype: Direct torch dtype (overrides dtype string if provided)
        **kwargs: Additional arguments passed to from_pretrained
        
    Returns:
        Loaded PreTrainedModel
        
    Example:
        >>> model = load_model("meta-llama/Llama-2-7b-hf", dtype="float16")
    """
    # Resolve dtype
    if torch_dtype is None:
        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        torch_dtype = dtype_map.get(dtype.lower() if dtype else "float16", torch.float16)
    
    logger.info(f"Loading model: {name_or_path}")
    logger.info(f"  dtype: {torch_dtype}, device_map: {device_map}")
    
    model = AutoModelForCausalLM.from_pretrained(
        name_or_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        **kwargs,
    )
    
    logger.info(f"Model loaded successfully: {model.__class__.__name__}")
    logger.info(f"  Parameters: {model.num_parameters():,}")
    
    return model


def load_tokenizer(
    name_or_path: str,
    trust_remote_code: bool = False,
    use_fast: bool = True,
    **kwargs,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    """Load a HuggingFace tokenizer.
    
    Args:
        name_or_path: Tokenizer name on HuggingFace Hub or local path
        trust_remote_code: Whether to trust remote code
        use_fast: Whether to use fast tokenizer if available
        **kwargs: Additional arguments passed to from_pretrained
        
    Returns:
        Loaded tokenizer
        
    Example:
        >>> tokenizer = load_tokenizer("meta-llama/Llama-2-7b-hf")
    """
    logger.info(f"Loading tokenizer: {name_or_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        name_or_path,
        trust_remote_code=trust_remote_code,
        use_fast=use_fast,
        **kwargs,
    )
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")
    
    return tokenizer

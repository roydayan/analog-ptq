"""Calibration data loading and handling."""

from typing import List, Optional, Union

import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from analog_ptq.utils.logging import get_logger


logger = get_logger(__name__)


class CalibrationDataLoader:
    """Handles loading and preparing calibration data for quantization.
    
    Supports loading from:
    - HuggingFace datasets (wikitext, c4, etc.)
    - Custom text data
    - Pre-tokenized tensors
    """
    
    SUPPORTED_DATASETS = {
        "wikitext": ("wikitext", "wikitext-2-raw-v1", "train", "text"),
        "wikitext-103": ("wikitext", "wikitext-103-raw-v1", "train", "text"),
        "c4": ("allenai/c4", "en", "train", "text"),
        "ptb": ("ptb_text_only", None, "train", "sentence"),
    }
    
    def __init__(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        seq_length: int = 2048,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """Initialize the calibration data loader.
        
        Args:
            tokenizer: Tokenizer for encoding text
            seq_length: Sequence length for calibration samples
            device: Device to place tensors on
        """
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.device = device or "cuda" if torch.cuda.is_available() else "cpu"
    
    def load_dataset_samples(
        self,
        dataset_name: str = "wikitext",
        num_samples: int = 128,
        seed: int = 42,
    ) -> List[torch.Tensor]:
        """Load calibration samples from a HuggingFace dataset.
        
        Args:
            dataset_name: Name of the dataset (wikitext, c4, etc.)
            num_samples: Number of calibration samples to generate
            seed: Random seed for reproducibility
            
        Returns:
            List of input_ids tensors
        """
        logger.info(f"Loading calibration data from {dataset_name}")
        
        if dataset_name in self.SUPPORTED_DATASETS:
            ds_name, ds_config, split, text_column = self.SUPPORTED_DATASETS[dataset_name]
        else:
            # Assume it's a direct HuggingFace dataset name
            ds_name = dataset_name
            ds_config = None
            split = "train"
            text_column = "text"
        
        # Load dataset
        dataset = load_dataset(
            ds_name,
            ds_config,
            split=split,
            trust_remote_code=True,
        )
        
        # Shuffle and take samples
        dataset = dataset.shuffle(seed=seed)
        
        # Concatenate all text
        all_text = []
        for item in dataset:
            text = item.get(text_column, "")
            if text and len(text.strip()) > 0:
                all_text.append(text)
            if len(all_text) >= num_samples * 10:  # Get extra for filtering
                break
        
        full_text = "\n\n".join(all_text)
        
        # Tokenize
        logger.info("Tokenizing calibration text...")
        encodings = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=False,
        )
        
        input_ids = encodings["input_ids"][0]
        
        # Split into samples
        samples = self._split_into_samples(input_ids, num_samples)
        
        logger.info(f"Created {len(samples)} calibration samples of length {self.seq_length}")
        
        return samples
    
    def load_from_text(
        self,
        texts: List[str],
        num_samples: Optional[int] = None,
    ) -> List[torch.Tensor]:
        """Load calibration samples from custom text data.
        
        Args:
            texts: List of text strings
            num_samples: Maximum number of samples (None for all)
            
        Returns:
            List of input_ids tensors
        """
        logger.info(f"Loading calibration data from {len(texts)} text samples")
        
        # Concatenate and tokenize
        full_text = "\n\n".join(texts)
        encodings = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=False,
        )
        
        input_ids = encodings["input_ids"][0]
        
        # Split into samples
        num_samples = num_samples or (len(input_ids) // self.seq_length)
        samples = self._split_into_samples(input_ids, num_samples)
        
        logger.info(f"Created {len(samples)} calibration samples")
        
        return samples
    
    def load_from_tensors(
        self,
        tensors: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Use pre-tokenized tensors as calibration data.
        
        Args:
            tensors: List of input_ids tensors
            
        Returns:
            Processed list of tensors
        """
        samples = []
        for tensor in tensors:
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            
            # Truncate or pad to seq_length
            if tensor.shape[-1] > self.seq_length:
                tensor = tensor[:, :self.seq_length]
            elif tensor.shape[-1] < self.seq_length:
                padding = torch.full(
                    (tensor.shape[0], self.seq_length - tensor.shape[-1]),
                    self.tokenizer.pad_token_id,
                    dtype=tensor.dtype,
                )
                tensor = torch.cat([tensor, padding], dim=-1)
            
            samples.append(tensor.to(self.device))
        
        return samples
    
    def _split_into_samples(
        self,
        input_ids: torch.Tensor,
        num_samples: int,
    ) -> List[torch.Tensor]:
        """Split a long sequence into fixed-length samples.
        
        Args:
            input_ids: 1D tensor of token ids
            num_samples: Number of samples to create
            
        Returns:
            List of sample tensors
        """
        total_length = len(input_ids)
        available_samples = total_length // self.seq_length
        
        if available_samples < num_samples:
            logger.warning(
                f"Only {available_samples} samples available, requested {num_samples}"
            )
            num_samples = available_samples
        
        samples = []
        for i in range(num_samples):
            start = i * self.seq_length
            end = start + self.seq_length
            sample = input_ids[start:end].unsqueeze(0).to(self.device)
            samples.append(sample)
        
        return samples
    
    def get_dataloader(
        self,
        samples: List[torch.Tensor],
        batch_size: int = 1,
    ):
        """Create a simple iterator over calibration samples.
        
        Args:
            samples: List of sample tensors
            batch_size: Batch size
            
        Yields:
            Batches of samples
        """
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            if len(batch) == 1:
                yield batch[0]
            else:
                yield torch.cat(batch, dim=0)

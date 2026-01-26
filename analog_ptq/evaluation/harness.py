"""Integration with lm-evaluation-harness for benchmarking."""

from typing import Any, Dict, List, Optional, Union

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from analog_ptq.models.wrapper import ModelWrapper
from analog_ptq.utils.logging import get_logger


logger = get_logger(__name__)


class LMEvalHarness:
    """Wrapper for lm-evaluation-harness to run benchmarks.
    
    Provides a simple interface to run standard LLM benchmarks
    on both regular and quantized models.
    
    Attributes:
        model: The model to evaluate
        tokenizer: Tokenizer for the model
    """
    
    # Common benchmark tasks
    STANDARD_TASKS = [
        "hellaswag",
        "arc_easy",
        "arc_challenge",
        "winogrande",
        "mmlu",
        "truthfulqa_mc",
    ]
    
    def __init__(
        self,
        model: Union[PreTrainedModel, ModelWrapper],
        tokenizer: PreTrainedTokenizer,
        device: Optional[str] = None,
    ):
        """Initialize the evaluation harness.
        
        Args:
            model: Model to evaluate
            tokenizer: Tokenizer for the model
            device: Device for evaluation (auto-detected if None)
        """
        if isinstance(model, ModelWrapper):
            self.model = model.model
        else:
            self.model = model
        
        self.tokenizer = tokenizer
        self.device = device or self._detect_device()
        
        self._lm = None  # Lazy-loaded lm-eval model wrapper
    
    def _detect_device(self) -> str:
        """Detect the device the model is on.
        
        Returns:
            Device string
        """
        try:
            param = next(self.model.parameters())
            return str(param.device)
        except StopIteration:
            return "cuda" if torch.cuda.is_available() else "cpu"
    
    def _get_lm_eval_model(self):
        """Get or create the lm-eval model wrapper.
        
        Returns:
            lm-eval compatible model
        """
        if self._lm is not None:
            return self._lm
        
        try:
            from lm_eval.models.huggingface import HFLM
        except ImportError:
            raise ImportError(
                "lm-eval is required for evaluation. "
                "Install with: pip install lm-eval"
            )
        
        self._lm = HFLM(
            pretrained=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
        )
        
        return self._lm
    
    def evaluate(
        self,
        tasks: Optional[List[str]] = None,
        num_fewshot: Optional[int] = None,
        batch_size: int = 8,
        limit: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run evaluation on specified tasks.
        
        Args:
            tasks: List of task names (uses standard tasks if None)
            num_fewshot: Number of few-shot examples
            batch_size: Batch size for evaluation
            limit: Limit number of samples per task (for quick testing)
            **kwargs: Additional arguments passed to lm-eval
            
        Returns:
            Dictionary containing evaluation results
        """
        try:
            from lm_eval import evaluator
            from lm_eval.tasks import TaskManager
        except ImportError:
            raise ImportError(
                "lm-eval is required for evaluation. "
                "Install with: pip install lm-eval"
            )
        
        tasks = tasks or self.STANDARD_TASKS[:3]  # Default to a few tasks
        
        logger.info(f"Running evaluation on tasks: {tasks}")
        logger.info(f"  batch_size={batch_size}, num_fewshot={num_fewshot}, limit={limit}")
        
        # Get the lm-eval model
        lm = self._get_lm_eval_model()
        
        # Initialize task manager
        task_manager = TaskManager()
        
        # Run evaluation
        results = evaluator.simple_evaluate(
            model=lm,
            tasks=tasks,
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            limit=limit,
            **kwargs,
        )
        
        # Log summary
        self._log_results_summary(results)
        
        return results
    
    def _log_results_summary(self, results: Dict[str, Any]) -> None:
        """Log a summary of evaluation results.
        
        Args:
            results: Evaluation results dictionary
        """
        logger.info("=" * 50)
        logger.info("Evaluation Results Summary")
        logger.info("=" * 50)
        
        if "results" not in results:
            logger.warning("No results found in output")
            return
        
        for task_name, task_results in results["results"].items():
            if isinstance(task_results, dict):
                # Find the main accuracy metric
                acc_key = None
                for key in ["acc", "acc_norm", "exact_match", "f1"]:
                    if key in task_results:
                        acc_key = key
                        break
                
                if acc_key:
                    acc = task_results[acc_key]
                    if isinstance(acc, (int, float)):
                        logger.info(f"  {task_name}: {acc:.4f} ({acc_key})")
                    else:
                        logger.info(f"  {task_name}: {acc} ({acc_key})")
                else:
                    logger.info(f"  {task_name}: {task_results}")
        
        logger.info("=" * 50)
    
    def quick_eval(
        self,
        tasks: Optional[List[str]] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """Run a quick evaluation with limited samples.
        
        Useful for testing and development.
        
        Args:
            tasks: List of task names
            limit: Number of samples per task
            
        Returns:
            Evaluation results
        """
        tasks = tasks or ["hellaswag"]
        
        logger.info(f"Running quick evaluation (limit={limit})")
        
        return self.evaluate(
            tasks=tasks,
            limit=limit,
            batch_size=4,
        )
    
    @staticmethod
    def list_available_tasks() -> List[str]:
        """List all available evaluation tasks.
        
        Returns:
            List of task names
        """
        try:
            from lm_eval.tasks import TaskManager
            task_manager = TaskManager()
            return list(task_manager.all_tasks)
        except ImportError:
            logger.warning("lm-eval not installed, returning common tasks")
            return LMEvalHarness.STANDARD_TASKS

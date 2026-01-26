"""Component registry for extensible quantization methods."""

from typing import Any, Callable, Dict, Optional, Type

from analog_ptq.utils.logging import get_logger


logger = get_logger(__name__)


class ComponentRegistry:
    """Registry for quantizers and other extensible components.
    
    Allows registration of custom quantization methods that can be
    referenced by name in configuration files.
    
    Example:
        >>> from analog_ptq.pipeline.registry import registry
        >>> 
        >>> @registry.register_quantizer("my_quantizer")
        >>> class MyQuantizer(BaseQuantizer):
        >>>     ...
        >>> 
        >>> # Later, in config:
        >>> # quantization:
        >>> #   method: "my_quantizer"
    """
    
    def __init__(self):
        """Initialize the registry."""
        self._quantizers: Dict[str, Type] = {}
        self._calibration_loaders: Dict[str, Type] = {}
        self._evaluators: Dict[str, Type] = {}
    
    def register_quantizer(
        self,
        name: str,
    ) -> Callable[[Type], Type]:
        """Register a quantizer class.
        
        Can be used as a decorator:
        
            @registry.register_quantizer("my_method")
            class MyQuantizer(BaseQuantizer):
                ...
        
        Args:
            name: Name to register the quantizer under
            
        Returns:
            Decorator function
        """
        def decorator(cls: Type) -> Type:
            if name in self._quantizers:
                logger.warning(f"Overwriting existing quantizer: {name}")
            self._quantizers[name] = cls
            logger.debug(f"Registered quantizer: {name}")
            return cls
        
        return decorator
    
    def register_calibration_loader(
        self,
        name: str,
    ) -> Callable[[Type], Type]:
        """Register a calibration data loader class.
        
        Args:
            name: Name to register the loader under
            
        Returns:
            Decorator function
        """
        def decorator(cls: Type) -> Type:
            if name in self._calibration_loaders:
                logger.warning(f"Overwriting existing loader: {name}")
            self._calibration_loaders[name] = cls
            logger.debug(f"Registered calibration loader: {name}")
            return cls
        
        return decorator
    
    def register_evaluator(
        self,
        name: str,
    ) -> Callable[[Type], Type]:
        """Register an evaluator class.
        
        Args:
            name: Name to register the evaluator under
            
        Returns:
            Decorator function
        """
        def decorator(cls: Type) -> Type:
            if name in self._evaluators:
                logger.warning(f"Overwriting existing evaluator: {name}")
            self._evaluators[name] = cls
            logger.debug(f"Registered evaluator: {name}")
            return cls
        
        return decorator
    
    def get_quantizer(self, name: str) -> Type:
        """Get a registered quantizer by name.
        
        Args:
            name: Quantizer name
            
        Returns:
            Quantizer class
            
        Raises:
            KeyError: If quantizer is not registered
        """
        if name not in self._quantizers:
            available = list(self._quantizers.keys())
            raise KeyError(
                f"Unknown quantizer: {name}. Available: {available}"
            )
        return self._quantizers[name]
    
    def get_calibration_loader(self, name: str) -> Type:
        """Get a registered calibration loader by name.
        
        Args:
            name: Loader name
            
        Returns:
            Loader class
        """
        if name not in self._calibration_loaders:
            available = list(self._calibration_loaders.keys())
            raise KeyError(
                f"Unknown calibration loader: {name}. Available: {available}"
            )
        return self._calibration_loaders[name]
    
    def get_evaluator(self, name: str) -> Type:
        """Get a registered evaluator by name.
        
        Args:
            name: Evaluator name
            
        Returns:
            Evaluator class
        """
        if name not in self._evaluators:
            available = list(self._evaluators.keys())
            raise KeyError(
                f"Unknown evaluator: {name}. Available: {available}"
            )
        return self._evaluators[name]
    
    def list_quantizers(self) -> list:
        """List all registered quantizers.
        
        Returns:
            List of quantizer names
        """
        return list(self._quantizers.keys())
    
    def list_calibration_loaders(self) -> list:
        """List all registered calibration loaders.
        
        Returns:
            List of loader names
        """
        return list(self._calibration_loaders.keys())
    
    def list_evaluators(self) -> list:
        """List all registered evaluators.
        
        Returns:
            List of evaluator names
        """
        return list(self._evaluators.keys())


# Global registry instance
registry = ComponentRegistry()


def register_quantizer(name: str) -> Callable[[Type], Type]:
    """Convenience function to register a quantizer.
    
    Args:
        name: Quantizer name
        
    Returns:
        Decorator function
    """
    return registry.register_quantizer(name)


def register_calibration_loader(name: str) -> Callable[[Type], Type]:
    """Convenience function to register a calibration loader.
    
    Args:
        name: Loader name
        
    Returns:
        Decorator function
    """
    return registry.register_calibration_loader(name)


def register_evaluator(name: str) -> Callable[[Type], Type]:
    """Convenience function to register an evaluator.
    
    Args:
        name: Evaluator name
        
    Returns:
        Decorator function
    """
    return registry.register_evaluator(name)


# Register built-in quantizers
def _register_builtins():
    """Register built-in quantizers."""
    from analog_ptq.quantization.gptq import GPTQQuantizer
    
    registry._quantizers["gptq"] = GPTQQuantizer


# Call at import time
_register_builtins()

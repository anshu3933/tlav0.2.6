# core/di/container.py

"""Dependency Injection container for managing service instances and dependencies."""

import inspect
from typing import Dict, Any, Optional, Type, TypeVar, Callable, get_type_hints
from config.logging_config import get_module_logger

# Create a logger for this module
logger = get_module_logger("di_container")

T = TypeVar('T')

class DIContainer:
    """Container for dependency injection and service management.
    
    This container manages service instantiation, lifetime, and dependencies.
    It allows for both eager and lazy initialization of services.
    """
    
    def __init__(self):
        """Initialize the container."""
        self._instances: Dict[str, Any] = {}
        self._factories: Dict[str, Callable[[], Any]] = {}
        self._types: Dict[str, Type] = {}
        
        logger.debug("Initialized dependency injection container")
    
    def register(self, interface_type: Type[T], implementation_type: Optional[Type] = None) -> None:
        """Register a type with the container.
        
        Args:
            interface_type: The interface or base type
            implementation_type: The concrete implementation type (defaults to interface_type)
        """
        if implementation_type is None:
            implementation_type = interface_type
            
        type_name = self._get_type_name(interface_type)
        self._types[type_name] = implementation_type
        
        logger.debug(f"Registered type: {type_name} -> {implementation_type.__name__}")
    
    def register_instance(self, interface_type: Type[T], instance: Any) -> None:
        """Register an existing instance with the container.
        
        Args:
            interface_type: The interface or base type
            instance: The instance to register
        """
        type_name = self._get_type_name(interface_type)
        self._instances[type_name] = instance
        
        logger.debug(f"Registered instance: {type_name} -> {instance}")
    
    def register_factory(self, interface_type: Type[T], factory: Callable[[], Any]) -> None:
        """Register a factory function for creating instances.
        
        Args:
            interface_type: The interface or base type
            factory: Factory function that creates instances
        """
        type_name = self._get_type_name(interface_type)
        self._factories[type_name] = factory
        
        logger.debug(f"Registered factory: {type_name}")
    
    def resolve(self, interface_type: Type[T]) -> T:
        """Resolve a type to an instance.
        
        Args:
            interface_type: The interface or base type to resolve
            
        Returns:
            An instance of the requested type
            
        Raises:
            KeyError: If the type is not registered
            Exception: If instance creation fails
        """
        type_name = self._get_type_name(interface_type)
        
        # Return existing instance if available
        if type_name in self._instances:
            return self._instances[type_name]
        
        # Use factory if registered
        if type_name in self._factories:
            instance = self._factories[type_name]()
            self._instances[type_name] = instance
            return instance
        
        # Create new instance from registered type
        if type_name in self._types:
            implementation_type = self._types[type_name]
            instance = self._create_instance(implementation_type)
            self._instances[type_name] = instance
            return instance
        
        raise KeyError(f"Type not registered: {type_name}")
    
    def _create_instance(self, implementation_type: Type) -> Any:
        """Create an instance with automatic dependency resolution.
        
        Args:
            implementation_type: The concrete implementation type
            
        Returns:
            An instance of the implementation type
            
        Raises:
            Exception: If instance creation fails
        """
        try:
            # Get constructor parameters
            init_params = inspect.signature(implementation_type.__init__).parameters
            
            # Skip self parameter
            params = list(init_params.items())
            if params and params[0][0] == 'self':
                params = params[1:]
            
            # Get parameter types from type hints
            type_hints = get_type_hints(implementation_type.__init__)
            
            # Build arguments dictionary
            kwargs = {}
            for name, param in params:
                # Skip *args and **kwargs
                if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                    continue
                
                # Use None for parameters without type hints
                if name not in type_hints:
                    if param.default is inspect.Parameter.empty:
                        kwargs[name] = None
                    continue
                
                param_type = type_hints[name]
                
                # Skip primitive types and parameters with default values
                if param_type in (str, int, float, bool, list, dict) or param.default is not inspect.Parameter.empty:
                    continue
                
                # Resolve the dependency
                try:
                    kwargs[name] = self.resolve(param_type)
                except KeyError:
                    # If the dependency is not registered and has a default value, skip it
                    if param.default is not inspect.Parameter.empty:
                        continue
                    logger.warning(f"Could not resolve dependency: {name} of type {param_type}")
                    raise
            
            # Create the instance
            return implementation_type(**kwargs)
            
        except Exception as e:
            logger.error(f"Error creating instance of {implementation_type.__name__}: {str(e)}")
            raise
    
    def _get_type_name(self, type_obj: Type) -> str:
        """Get a unique name for a type.
        
        Args:
            type_obj: The type to get a name for
            
        Returns:
            A unique name for the type
        """
        if hasattr(type_obj, "__name__"):
            return type_obj.__name__
        else:
            # Handle TypeVar and similar types
            return str(type_obj)
    
    def clear(self) -> None:
        """Clear all registered instances and factories."""
        self._instances.clear()
        self._factories.clear()
        self._types.clear()
        
        logger.debug("Cleared dependency injection container")


# Create a singleton instance
container = DIContainer()
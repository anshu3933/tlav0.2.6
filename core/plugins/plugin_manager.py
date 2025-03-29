# core/plugins/plugin_manager.py

"""Plugin management system for extensibility."""

import os
import sys
import importlib
import inspect
import json
import pkgutil
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Type, Callable, TypeVar
from dataclasses import dataclass, field
from config.logging_config import get_module_logger
from core.events.typed_event_bus import Event, event_bus

# Create a logger for this module
logger = get_module_logger("plugin_manager")

# Type variable for plugin types
T = TypeVar('T')

@dataclass
class PluginMetadata:
    """Metadata for a plugin."""
    name: str
    version: str
    description: str
    author: str = ""
    dependencies: List[str] = field(default_factory=list)
    enabled: bool = True
    entry_point: str = ""


class Plugin(ABC):
    """Base class for all plugins."""
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the plugin.
        
        Returns:
            True if initialization succeeded, False otherwise
        """
        pass
    
    @abstractmethod
    def shutdown(self) -> bool:
        """Shutdown the plugin.
        
        Returns:
            True if shutdown succeeded, False otherwise
        """
        pass
    
    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Get plugin metadata.
        
        Returns:
            Plugin metadata
        """
        pass


@Event
class PluginLoadedEvent:
    """Event fired when a plugin is loaded."""
    plugin_name: str
    plugin_version: str
    timestamp: str


@Event
class PluginUnloadedEvent:
    """Event fired when a plugin is unloaded."""
    plugin_name: str
    timestamp: str


class PluginManager:
    """Manages plugins and extensibility.
    
    This manager handles plugin discovery, loading, and lifecycle management.
    """
    
    def __init__(self, plugins_dir: str = "plugins"):
        """Initialize the plugin manager.
        
        Args:
            plugins_dir: Directory containing plugins
        """
        self.plugins_dir = plugins_dir
        self._plugins: Dict[str, Plugin] = {}
        self._plugin_classes: Dict[str, Type[Plugin]] = {}
        self._hooks: Dict[str, List[Callable]] = {}
        
        # Ensure plugins directory exists
        os.makedirs(plugins_dir, exist_ok=True)
        
        logger.debug(f"Initialized plugin manager with directory: {plugins_dir}")
    
    def discover_plugins(self) -> List[PluginMetadata]:
        """Discover available plugins.
        
        Returns:
            List of plugin metadata
        """
        discovered_plugins = []
        
        # Add plugins directory to path
        sys.path.insert(0, self.plugins_dir)
        
        try:
            # Find plugin packages
            for _, name, ispkg in pkgutil.iter_modules([self.plugins_dir]):
                if ispkg:
                    try:
                        # Check for plugin metadata
                        metadata_path = os.path.join(self.plugins_dir, name, "plugin.json")
                        if os.path.exists(metadata_path):
                            with open(metadata_path, "r") as f:
                                metadata_dict = json.load(f)
                                metadata = PluginMetadata(
                                    name=metadata_dict.get("name", name),
                                    version=metadata_dict.get("version", "0.1.0"),
                                    description=metadata_dict.get("description", ""),
                                    author=metadata_dict.get("author", ""),
                                    dependencies=metadata_dict.get("dependencies", []),
                                    enabled=metadata_dict.get("enabled", True),
                                    entry_point=metadata_dict.get("entry_point", "")
                                )
                                discovered_plugins.append(metadata)
                                
                                logger.debug(f"Discovered plugin: {metadata.name} v{metadata.version}")
                    except Exception as e:
                        logger.error(f"Error discovering plugin {name}: {str(e)}")
                        continue
        finally:
            # Remove plugins directory from path
            if self.plugins_dir in sys.path:
                sys.path.remove(self.plugins_dir)
        
        return discovered_plugins
    
    def load_plugin(self, plugin_name: str) -> Optional[Plugin]:
        """Load a plugin by name.
        
        Args:
            plugin_name: Name of the plugin to load
            
        Returns:
            Loaded plugin or None if loading failed
        """
        # Check if plugin is already loaded
        if plugin_name in self._plugins:
            logger.debug(f"Plugin already loaded: {plugin_name}")
            return self._plugins[plugin_name]
        
        # Add plugins directory to path
        sys.path.insert(0, self.plugins_dir)
        
        try:
            # Load plugin module
            plugin_module = importlib.import_module(f"{plugin_name}")
            
            # Look for plugin metadata
            metadata_path = os.path.join(self.plugins_dir, plugin_name, "plugin.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata_dict = json.load(f)
                    entry_point = metadata_dict.get("entry_point", "")
            else:
                # Default entry point based on package name
                entry_point = f"{plugin_name}.plugin.Plugin"
            
            # Find plugin class
            parts = entry_point.split(".")
            module_path, class_name = ".".join(parts[:-1]), parts[-1]
            
            if module_path:
                plugin_class_module = importlib.import_module(module_path)
                plugin_class = getattr(plugin_class_module, class_name)
            else:
                plugin_class = getattr(plugin_module, class_name)
            
            # Instantiate plugin
            plugin = plugin_class()
            
            # Initialize plugin
            if not plugin.initialize():
                logger.error(f"Failed to initialize plugin: {plugin_name}")
                return None
            
            # Store plugin
            self._plugins[plugin_name] = plugin
            self._plugin_classes[plugin_name] = plugin_class
            
            # Publish event
            event_bus.publish(PluginLoadedEvent(
                plugin_name=plugin.metadata.name,
                plugin_version=plugin.metadata.version,
                timestamp=__import__('datetime').datetime.now().isoformat()
            ))
            
            logger.info(f"Loaded plugin: {plugin.metadata.name} v{plugin.metadata.version}")
            return plugin
            
        except Exception as e:
            logger.error(f"Error loading plugin {plugin_name}: {str(e)}", exc_info=True)
            return None
        finally:
            # Remove plugins directory from path
            if self.plugins_dir in sys.path:
                sys.path.remove(self.plugins_dir)
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin.
        
        Args:
            plugin_name: Name of the plugin to unload
            
        Returns:
            True if unloaded successfully, False otherwise
        """
        # Check if plugin is loaded
        if plugin_name not in self._plugins:
            logger.warning(f"Plugin not loaded: {plugin_name}")
            return False
        
        plugin = self._plugins[plugin_name]
        
        try:
            # Shutdown plugin
            if not plugin.shutdown():
                logger.error(f"Failed to shutdown plugin: {plugin_name}")
                return False
            
            # Remove plugin
            del self._plugins[plugin_name]
            if plugin_name in self._plugin_classes:
                del self._plugin_classes[plugin_name]
            
            # Publish event
            event_bus.publish(PluginUnloadedEvent(
                plugin_name=plugin.metadata.name,
                timestamp=__import__('datetime').datetime.now().isoformat()
            ))
            
            logger.info(f"Unloaded plugin: {plugin.metadata.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error unloading plugin {plugin_name}: {str(e)}", exc_info=True)
            return False
    
    def get_plugin(self, plugin_name: str) -> Optional[Plugin]:
        """Get a loaded plugin by name.
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            Plugin or None if not loaded
        """
        return self._plugins.get(plugin_name)
    
    def get_plugins(self) -> Dict[str, Plugin]:
        """Get all loaded plugins.
        
        Returns:
            Dictionary of plugin name to plugin instance
        """
        return dict(self._plugins)
    
    def register_hook(self, hook_name: str, callback: Callable) -> bool:
        """Register a hook callback.
        
        Args:
            hook_name: Name of the hook
            callback: Callback function
            
        Returns:
            True if registered successfully, False otherwise
        """
        if hook_name not in self._hooks:
            self._hooks[hook_name] = []
        
        self._hooks[hook_name].append(callback)
        logger.debug(f"Registered hook '{hook_name}': {callback.__name__}")
        return True
    
    def unregister_hook(self, hook_name: str, callback: Callable) -> bool:
        """Unregister a hook callback.
        
        Args:
            hook_name: Name of the hook
            callback: Callback function
            
        Returns:
            True if unregistered successfully, False otherwise
        """
        if hook_name not in self._hooks:
            logger.warning(f"Hook not found: {hook_name}")
            return False
        
        if callback not in self._hooks[hook_name]:
            logger.warning(f"Callback not registered for hook: {hook_name}")
            return False
        
        self._hooks[hook_name].remove(callback)
        logger.debug(f"Unregistered hook '{hook_name}': {callback.__name__}")
        return True
    
    def call_hook(self, hook_name: str, *args, **kwargs) -> List[Any]:
        """Call all callbacks for a hook.
        
        Args:
            hook_name: Name of the hook
            *args: Positional arguments to pass to callbacks
            **kwargs: Keyword arguments to pass to callbacks
            
        Returns:
            List of results from callbacks
        """
        if hook_name not in self._hooks:
            return []
        
        results = []
        
        for callback in self._hooks[hook_name]:
            try:
                result = callback(*args, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Error in hook '{hook_name}' callback {callback.__name__}: {str(e)}")
        
        logger.debug(f"Called hook '{hook_name}' with {len(results)} results")
        return results
    
    def get_plugin_instance(self, plugin_type: Type[T]) -> Optional[T]:
        """Get a plugin instance by type.
        
        Args:
            plugin_type: Type of plugin to find
            
        Returns:
            Plugin instance or None if not found
        """
        for plugin in self._plugins.values():
            if isinstance(plugin, plugin_type):
                return plugin
        
        return None
    
    def get_plugin_instances(self, plugin_type: Type[T]) -> List[T]:
        """Get all plugin instances of a type.
        
        Args:
            plugin_type: Type of plugins to find
            
        Returns:
            List of plugin instances
        """
        return [
            plugin for plugin in self._plugins.values()
            if isinstance(plugin, plugin_type)
        ]


# Create a singleton instance
plugin_manager = PluginManager()
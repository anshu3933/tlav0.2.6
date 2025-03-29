# core/plugins/test_plugin_manager.py

"""Tests for the plugin manager implementation."""

import os
import unittest
import tempfile
import shutil
from typing import Dict, Any, List
from dataclasses import dataclass, field
from core.plugins.plugin_manager import Plugin, PluginManager, PluginMetadata
from config.logging_config import get_module_logger

# Create a logger for this module
logger = get_module_logger("test_plugin_manager")

class TestPlugin(Plugin):
    """Test plugin for plugin manager testing."""
    
    def __init__(self, name="Test Plugin", version="1.0.0"):
        """Initialize with name and version."""
        self._metadata = PluginMetadata(
            name=name,
            version=version,
            description="A test plugin for testing",
            author="Tester"
        )
        self.initialized = False
        self.shutdown_called = False
    
    def initialize(self) -> bool:
        """Initialize the plugin."""
        self.initialized = True
        return True
    
    def shutdown(self) -> bool:
        """Shutdown the plugin."""
        self.shutdown_called = True
        return True
    
    @property
    def metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        return self._metadata

class ErrorPlugin(Plugin):
    """Plugin that raises errors during lifecycle methods."""
    
    def __init__(self):
        """Initialize the plugin."""
        self._metadata = PluginMetadata(
            name="Error Plugin",
            version="1.0.0",
            description="A plugin that raises errors",
            author="Tester"
        )
    
    def initialize(self) -> bool:
        """Initialize the plugin with error."""
        raise ValueError("Initialization error")
    
    def shutdown(self) -> bool:
        """Shutdown the plugin with error."""
        raise ValueError("Shutdown error")
    
    @property
    def metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        return self._metadata

class HookTestPlugin(Plugin):
    """Plugin for testing hooks."""
    
    def __init__(self):
        """Initialize the plugin."""
        self._metadata = PluginMetadata(
            name="Hook Test Plugin",
            version="1.0.0",
            description="A plugin for testing hooks",
            author="Tester"
        )
        self.hook_called = False
        self.hook_args = None
        self.hook_kwargs = None
    
    def initialize(self) -> bool:
        """Initialize the plugin."""
        return True
    
    def shutdown(self) -> bool:
        """Shutdown the plugin."""
        return True
    
    @property
    def metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        return self._metadata
    
    def test_hook(self, *args, **kwargs):
        """Test hook method."""
        self.hook_called = True
        self.hook_args = args
        self.hook_kwargs = kwargs
        return "hook_result"

class PluginManagerTests(unittest.TestCase):
    """Test cases for the plugin manager."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary plugins directory
        self.test_dir = tempfile.mkdtemp()
        self.manager = PluginManager(plugins_dir=self.test_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_plugin_lifecycle(self):
        """Test plugin loading and unloading."""
        # Create a test plugin
        plugin = TestPlugin()
        
        # Register plugin class
        self.manager._plugin_classes["test_plugin"] = TestPlugin
        self.manager._plugins["test_plugin"] = plugin
        
        # Get plugin
        retrieved_plugin = self.manager.get_plugin("test_plugin")
        self.assertEqual(retrieved_plugin, plugin)
        
        # Unload plugin
        result = self.manager.unload_plugin("test_plugin")
        self.assertTrue(result)
        self.assertTrue(plugin.shutdown_called)
        
        # Check that plugin is removed
        self.assertIsNone(self.manager.get_plugin("test_plugin"))
    
    def test_error_handling(self):
        """Test error handling during plugin lifecycle."""
        # Create an error plugin
        plugin = ErrorPlugin()
        
        # Register plugin class
        self.manager._plugin_classes["error_plugin"] = ErrorPlugin
        
        # Try to load plugin (should fail)
        loaded_plugin = self.manager.load_plugin("error_plugin")
        self.assertIsNone(loaded_plugin)
        
        # Add the plugin directly to test unloading
        self.manager._plugins["error_plugin"] = plugin
        
        # Try to unload plugin (should handle error)
        result = self.manager.unload_plugin("error_plugin")
        self.assertFalse(result)
    
    def test_hooks(self):
        """Test hook registration and calling."""
        # Create a hook test plugin
        plugin = HookTestPlugin()
        
        # Register plugin
        self.manager._plugins["hook_plugin"] = plugin
        
        # Register hook
        self.manager.register_hook("test_hook", plugin.test_hook)
        
        # Call hook
        results = self.manager.call_hook("test_hook", "arg1", "arg2", kwarg1="value1")
        
        # Check that hook was called
        self.assertTrue(plugin.hook_called)
        self.assertEqual(plugin.hook_args, ("arg1", "arg2"))
        self.assertEqual(plugin.hook_kwargs, {"kwarg1": "value1"})
        
        # Check that result was returned
        self.assertEqual(results, ["hook_result"])
        
        # Unregister hook
        result = self.manager.unregister_hook("test_hook", plugin.test_hook)
        self.assertTrue(result)
        
        # Call hook again
        results = self.manager.call_hook("test_hook")
        
        # Check that hook was not called again
        self.assertEqual(len(results), 0)
    
    def test_multiple_hooks(self):
        """Test multiple hooks for the same hook name."""
        # Create two plugins
        plugin1 = HookTestPlugin()
        plugin2 = HookTestPlugin()
        
        # Register plugins
        self.manager._plugins["hook_plugin1"] = plugin1
        self.manager._plugins["hook_plugin2"] = plugin2
        
        # Register hooks
        self.manager.register_hook("shared_hook", plugin1.test_hook)
        self.manager.register_hook("shared_hook", plugin2.test_hook)
        
        # Call hook
        results = self.manager.call_hook("shared_hook")
        
        # Check that both hooks were called
        self.assertTrue(plugin1.hook_called)
        self.assertTrue(plugin2.hook_called)
        
        # Check that results were returned
        self.assertEqual(results, ["hook_result", "hook_result"])
    
    def test_hook_error_handling(self):
        """Test error handling in hooks."""
        # Define a hook that raises an exception
        def error_hook():
            raise ValueError("Hook error")
        
        # Define a success hook
        success_results = []
        def success_hook():
            success_results.append("called")
            return "success"
        
        # Register hooks
        self.manager.register_hook("error_test", error_hook)
        self.manager.register_hook("error_test", success_hook)
        
        # Call hook - the error in one hook should not affect the other
        results = self.manager.call_hook("error_test")
        
        # Check that the success hook was still called
        self.assertEqual(success_results, ["called"])
        self.assertEqual(results, ["success"])
    
    def test_plugin_discovery(self):
        """Test plugin discovery in plugins directory."""
        # Create a test plugin directory
        plugin_dir = os.path.join(self.test_dir, "test_plugin")
        os.makedirs(plugin_dir)
        
        # Create a plugin.json file
        plugin_json = {
            "name": "Test Plugin",
            "version": "1.0.0",
            "description": "A test plugin",
            "author": "Tester",
            "entry_point": "test_plugin.plugin.Plugin"
        }
        
        with open(os.path.join(plugin_dir, "plugin.json"), "w") as f:
            import json
            json.dump(plugin_json, f)
        
        # Create an __init__.py file
        with open(os.path.join(plugin_dir, "__init__.py"), "w") as f:
            f.write("# Test plugin package")
        
        # Discover plugins
        discovered = self.manager.discover_plugins()
        
        # Check that the plugin was discovered
        self.assertEqual(len(discovered), 1)
        self.assertEqual(discovered[0].name, "Test Plugin")
        self.assertEqual(discovered[0].version, "1.0.0")
    
    def test_get_plugin_instance(self):
        """Test getting plugin instances by type."""
        # Create and register plugins
        plugin1 = TestPlugin("Plugin 1")
        plugin2 = TestPlugin("Plugin 2")
        plugin3 = HookTestPlugin()
        
        self.manager._plugins["plugin1"] = plugin1
        self.manager._plugins["plugin2"] = plugin2
        self.manager._plugins["plugin3"] = plugin3
        
        # Get plugin instance by type
        instance = self.manager.get_plugin_instance(TestPlugin)
        self.assertIsInstance(instance, TestPlugin)
        
        # Get all instances of a type
        instances = self.manager.get_plugin_instances(TestPlugin)
        self.assertEqual(len(instances), 2)
        self.assertIn(plugin1, instances)
        self.assertIn(plugin2, instances)
        
        # Get instance of another type
        hook_instance = self.manager.get_plugin_instance(HookTestPlugin)
        self.assertEqual(hook_instance, plugin3)

def run_tests():
    """Run the plugin manager tests."""
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

if __name__ == "__main__":
    run_tests()
# core/events/test_typed_event_bus.py

"""Tests for the typed event bus implementation."""

import time
import unittest
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional
from core.events.typed_event_bus import Event, EventBus, EventSubscription
from config.logging_config import get_module_logger

# Create a logger for this module
logger = get_module_logger("test_event_bus")

@dataclass
class TestEvent(Event):
    """Test event for event bus testing."""
    message: str
    value: int

@dataclass
class SpecializedTestEvent(TestEvent):
    """Specialized test event that extends TestEvent."""
    special_value: str

class TypedEventBusTests(unittest.TestCase):
    """Test cases for the typed event bus."""
    
    def setUp(self):
        """Set up test environment."""
        self.event_bus = EventBus()
        self.received_events = []
        self.received_specialized_events = []
    
    def handle_test_event(self, event: TestEvent):
        """Handle test events for testing."""
        self.received_events.append(event)
    
    def handle_specialized_event(self, event: SpecializedTestEvent):
        """Handle specialized test events for testing."""
        self.received_specialized_events.append(event)
    
    def test_simple_subscribe_publish(self):
        """Test basic subscription and publishing."""
        # Subscribe to test events
        subscription = self.event_bus.subscribe(TestEvent, self.handle_test_event)
        
        # Publish a test event
        test_event = TestEvent(message="Test Message", value=42)
        self.event_bus.publish(test_event)
        
        # Check that the event was received
        self.assertEqual(len(self.received_events), 1)
        self.assertEqual(self.received_events[0].message, "Test Message")
        self.assertEqual(self.received_events[0].value, 42)
        
        # Unsubscribe
        self.event_bus.unsubscribe(subscription)
        
        # Publish another event
        self.event_bus.publish(TestEvent(message="Second Message", value=100))
        
        # Check that no new event was received
        self.assertEqual(len(self.received_events), 1)
    
    def test_inheritance_subscription(self):
        """Test subscription with inheritance."""
        # Subscribe to base event type
        base_subscription = self.event_bus.subscribe(TestEvent, self.handle_test_event)
        
        # Subscribe to specialized event type
        specialized_subscription = self.event_bus.subscribe(SpecializedTestEvent, self.handle_specialized_event)
        
        # Publish a specialized event
        specialized_event = SpecializedTestEvent(message="Specialized", value=99, special_value="Special")
        self.event_bus.publish(specialized_event)
        
        # Check that both handlers received the event
        self.assertEqual(len(self.received_events), 1)
        self.assertEqual(self.received_events[0].message, "Specialized")
        
        self.assertEqual(len(self.received_specialized_events), 1)
        self.assertEqual(self.received_specialized_events[0].special_value, "Special")
        
        # Publish a base event
        base_event = TestEvent(message="Base", value=50)
        self.event_bus.publish(base_event)
        
        # Check that only the base handler received it
        self.assertEqual(len(self.received_events), 2)
        self.assertEqual(len(self.received_specialized_events), 1)
    
    def test_async_event_bus(self):
        """Test asynchronous event bus."""
        # Create async event bus with 2 workers
        async_bus = EventBus(async_mode=True, worker_count=2)
        
        # Storage for received events
        received = []
        event_received = threading.Event()
        
        # Handler with artificial delay
        def slow_handler(event: TestEvent):
            time.sleep(0.1)  # Simulate processing time
            received.append(event)
            event_received.set()
        
        # Subscribe to events
        async_bus.subscribe(TestEvent, slow_handler)
        
        # Publish an event
        start_time = time.time()
        async_bus.publish(TestEvent(message="Async", value=42))
        
        # Check that the call returns immediately
        publish_time = time.time() - start_time
        self.assertLess(publish_time, 0.05)  # Should return almost immediately
        
        # Wait for the event to be processed
        event_received.wait(timeout=1.0)
        
        # Check that the event was received
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].message, "Async")
        
        # Shut down the event bus
        async_bus.shutdown()
    
    def test_multiple_subscriptions(self):
        """Test multiple subscriptions to the same event type."""
        # Storage for different handlers
        handler1_events = []
        handler2_events = []
        
        # Define handlers
        def handler1(event: TestEvent):
            handler1_events.append(event)
        
        def handler2(event: TestEvent):
            handler2_events.append(event)
        
        # Subscribe both handlers
        sub1 = self.event_bus.subscribe(TestEvent, handler1)
        sub2 = self.event_bus.subscribe(TestEvent, handler2)
        
        # Publish an event
        self.event_bus.publish(TestEvent(message="Multiple", value=123))
        
        # Check that both handlers received the event
        self.assertEqual(len(handler1_events), 1)
        self.assertEqual(len(handler2_events), 1)
        
        # Unsubscribe one handler
        self.event_bus.unsubscribe(sub1)
        
        # Publish another event
        self.event_bus.publish(TestEvent(message="After Unsub", value=456))
        
        # Check that only handler2 received the event
        self.assertEqual(len(handler1_events), 1)
        self.assertEqual(len(handler2_events), 2)
    
    def test_error_handling(self):
        """Test error handling in event handlers."""
        # Define handler that raises an exception
        def error_handler(event: TestEvent):
            raise ValueError("Test error")
        
        # Storage for successful handler
        success_events = []
        def success_handler(event: TestEvent):
            success_events.append(event)
        
        # Subscribe both handlers
        self.event_bus.subscribe(TestEvent, error_handler)
        self.event_bus.subscribe(TestEvent, success_handler)
        
        # Publish an event - the error in one handler should not affect the other
        self.event_bus.publish(TestEvent(message="Error Test", value=789))
        
        # Check that the successful handler still received the event
        self.assertEqual(len(success_events), 1)
        self.assertEqual(success_events[0].message, "Error Test")
    
    def test_invalid_event_type(self):
        """Test handling of invalid event types."""
        # Try to subscribe to a non-Event class
        with self.assertRaises(TypeError):
            self.event_bus.subscribe(str, lambda s: None)
        
        # Try to publish a non-Event object
        with self.assertRaises(TypeError):
            self.event_bus.publish("Not an event")

def run_tests():
    """Run the event bus tests."""
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

if __name__ == "__main__":
    run_tests()
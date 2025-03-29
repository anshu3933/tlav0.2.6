# core/events/typed_event_bus.py

"""Typed event bus for event-driven architecture with strong typing."""

import threading
import queue
import time
import uuid
from dataclasses import dataclass, field, is_dataclass
from typing import Dict, List, Any, Type, TypeVar, Generic, Callable, Optional, get_type_hints
from datetime import datetime
from config.logging_config import get_module_logger

# Create a logger for this module
logger = get_module_logger("typed_event_bus")

T = TypeVar('T')

@dataclass
class Event:
    """Base class for all events."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class EventSubscription(Generic[T]):
    """Subscription to a specific event type."""
    
    def __init__(self, 
                 event_type: Type[T], 
                 handler: Callable[[T], None], 
                 subscription_id: str = None):
        """Initialize with event type and handler.
        
        Args:
            event_type: The event type to subscribe to
            handler: Handler function that accepts the event
            subscription_id: Optional unique ID for the subscription
        """
        self.event_type = event_type
        self.handler = handler
        self.subscription_id = subscription_id or str(uuid.uuid4())


class EventBus:
    """Typed event bus that supports publishing and subscribing to events.
    
    This implementation supports both synchronous and asynchronous event processing.
    """
    
    def __init__(self, async_mode: bool = False, worker_count: int = 1):
        """Initialize the event bus.
        
        Args:
            async_mode: Whether to process events asynchronously
            worker_count: Number of worker threads for async processing
        """
        self._subscriptions: Dict[Type, List[EventSubscription]] = {}
        self._lock = threading.RLock()
        self._async_mode = async_mode
        self._worker_count = worker_count
        self._event_queue = queue.Queue() if async_mode else None
        self._workers = []
        self._running = False
        
        if async_mode:
            self._start_workers()
        
        logger.debug(f"Initialized event bus (async_mode={async_mode}, workers={worker_count if async_mode else 0})")
    
    def _start_workers(self) -> None:
        """Start worker threads for asynchronous event processing."""
        self._running = True
        
        for i in range(self._worker_count):
            worker = threading.Thread(target=self._worker_loop, name=f"EventBusWorker-{i}")
            worker.daemon = True  # Allow the thread to exit when the main program exits
            worker.start()
            self._workers.append(worker)
            
        logger.debug(f"Started {self._worker_count} event bus worker threads")
    
    def _worker_loop(self) -> None:
        """Worker thread loop for processing events."""
        while self._running:
            try:
                # Get the next event from the queue, with timeout
                event_type, event = self._event_queue.get(timeout=0.1)
                
                # Process the event
                self._process_event(event_type, event)
                
                # Mark the task as done
                self._event_queue.task_done()
                
            except queue.Empty:
                # No events in the queue, just continue
                continue
                
            except Exception as e:
                logger.error(f"Error in event bus worker: {str(e)}", exc_info=True)
    
    def subscribe(self, event_type: Type[T], handler: Callable[[T], None]) -> EventSubscription:
        """Subscribe to a specific event type.
        
        Args:
            event_type: The event type to subscribe to
            handler: Handler function that accepts the event
            
        Returns:
            The event subscription
        """
        if not (is_dataclass(event_type) and issubclass(event_type, Event)):
            raise TypeError(f"Event type must be a dataclass derived from Event: {event_type}")
        
        subscription = EventSubscription(event_type, handler)
        
        with self._lock:
            if event_type not in self._subscriptions:
                self._subscriptions[event_type] = []
            
            self._subscriptions[event_type].append(subscription)
        
        logger.debug(f"Added subscription for {event_type.__name__}: {subscription.subscription_id}")
        return subscription
    
    def unsubscribe(self, subscription: EventSubscription) -> bool:
        """Unsubscribe from an event.
        
        Args:
            subscription: The subscription to remove
            
        Returns:
            True if the subscription was removed, False otherwise
        """
        event_type = subscription.event_type
        
        with self._lock:
            if event_type in self._subscriptions:
                subscriptions = self._subscriptions[event_type]
                
                for i, sub in enumerate(subscriptions):
                    if sub.subscription_id == subscription.subscription_id:
                        subscriptions.pop(i)
                        logger.debug(f"Removed subscription: {subscription.subscription_id}")
                        
                        # Remove the event type if there are no more subscriptions
                        if not subscriptions:
                            del self._subscriptions[event_type]
                        
                        return True
        
        return False
    
    def publish(self, event: T) -> None:
        """Publish an event to all subscribers.
        
        Args:
            event: The event to publish
        """
        if not isinstance(event, Event):
            raise TypeError(f"Event must be derived from Event: {type(event)}")
        
        event_type = type(event)
        
        if self._async_mode:
            # Add to queue for async processing
            self._event_queue.put((event_type, event))
        else:
            # Process synchronously
            self._process_event(event_type, event)
    
    def _process_event(self, event_type: Type[T], event: T) -> None:
        """Process an event by calling all subscribed handlers.
        
        Args:
            event_type: The type of the event
            event: The event to process
        """
        # Get all handlers for this event type and its base classes
        handlers = []
        
        with self._lock:
            # Find handlers for this event type
            if event_type in self._subscriptions:
                handlers.extend(self._subscriptions[event_type])
            
            # Find handlers for base classes
            for base_type, subs in self._subscriptions.items():
                if event_type != base_type and issubclass(event_type, base_type):
                    handlers.extend(subs)
        
        # Call all handlers
        for subscription in handlers:
            try:
                subscription.handler(event)
            except Exception as e:
                logger.error(f"Error in event handler for {event_type.__name__}: {str(e)}", exc_info=True)
    
    def shutdown(self) -> None:
        """Shutdown the event bus and worker threads."""
        if self._async_mode:
            # Signal workers to stop
            self._running = False
            
            # Wait for workers to finish
            for worker in self._workers:
                if worker.is_alive():
                    worker.join(timeout=1.0)
            
            # Clear worker list
            self._workers = []
        
        # Clear subscriptions
        with self._lock:
            self._subscriptions.clear()
        
        logger.debug("Event bus shutdown complete")


# Create a singleton instance
event_bus = EventBus(async_mode=True, worker_count=2)
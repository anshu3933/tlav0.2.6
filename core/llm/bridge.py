# core/llm/bridge.py

"""Bridge for abstracting interactions with different LLM providers."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Callable
import time
import os
import uuid
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from config.logging_config import get_module_logger
from core.events.typed_event_bus import Event, event_bus

# Create a logger for this module
logger = get_module_logger("llm_bridge")

class ModelProvider(Enum):
    """Supported model providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    VERTEX = "vertex"
    LLAMA = "llama"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"

@dataclass
class LLMRequest:
    """Request to a language model."""
    prompt: str
    system_prompt: Optional[str] = None
    messages: List[Dict[str, str]] = field(default_factory=list)
    temperature: float = 0.7
    max_tokens: int = 2000
    stop_sequences: List[str] = field(default_factory=list)
    top_p: float = 1.0
    top_k: int = 40
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    user_id: Optional[str] = None
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    request_type: str = "chat"  # chat, completion, embedding

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prompt": self.prompt,
            "system_prompt": self.system_prompt,
            "messages": self.messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stop_sequences": self.stop_sequences,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "user_id": self.user_id,
            "request_id": self.request_id,
            "timestamp": self.timestamp,
            "request_type": self.request_type
        }

@dataclass
class LLMResponse:
    """Response from a language model."""
    text: str
    request_id: str
    completion_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    provider: str = "unknown"
    model: str = "unknown"
    usage: Dict[str, int] = field(default_factory=dict)
    raw_response: Any = None
    finish_reason: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    latency_ms: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "request_id": self.request_id,
            "completion_id": self.completion_id,
            "provider": self.provider,
            "model": self.model,
            "usage": self.usage,
            "finish_reason": self.finish_reason,
            "timestamp": self.timestamp,
            "latency_ms": self.latency_ms,
            "error": self.error
        }

    @property
    def is_success(self) -> bool:
        """Check if response was successful."""
        return self.error is None

@Event
class LLMRequestEvent:
    """Event fired before an LLM request is made."""
    request_id: str
    provider: str
    model: str
    request_type: str
    timestamp: str

@Event
class LLMResponseEvent:
    """Event fired after an LLM response is received."""
    request_id: str
    completion_id: str
    provider: str
    model: str
    is_success: bool
    error: Optional[str]
    latency_ms: int
    token_usage: Dict[str, int]
    timestamp: str

class LLMBridge(ABC):
    """Base class for LLM provider bridges."""
    
    def __init__(self, model: str, provider: ModelProvider):
        """Initialize with model and provider.
        
        Args:
            model: Model name
            provider: Model provider
        """
        self.model = model
        self.provider = provider
        logger.debug(f"Initialized {provider.value} bridge with model {model}")
    
    @abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate text from a prompt.
        
        Args:
            request: LLM request
            
        Returns:
            LLM response
        """
        pass
    
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        pass
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate a response in a chat context.
        
        Args:
            messages: List of message dictionaries with role and content
            **kwargs: Additional parameters
            
        Returns:
            LLM response
        """
        # Extract system prompt if present
        system_prompt = None
        filtered_messages = []
        
        for message in messages:
            if message.get("role") == "system":
                system_prompt = message.get("content", "")
            else:
                filtered_messages.append(message)
        
        # Create prompt from the last user message
        prompt = next((m.get("content", "") for m in reversed(filtered_messages) 
                      if m.get("role") == "user"), "")
        
        # Create request
        request = LLMRequest(
            prompt=prompt,
            system_prompt=system_prompt,
            messages=filtered_messages,
            request_type="chat",
            **kwargs
        )
        
        # Generate response
        return self.generate(request)
    
    def _emit_request_event(self, request: LLMRequest):
        """Emit an LLM request event.
        
        Args:
            request: LLM request
        """
        event_bus.publish(LLMRequestEvent(
            request_id=request.request_id,
            provider=self.provider.value,
            model=self.model,
            request_type=request.request_type,
            timestamp=datetime.now().isoformat()
        ))
    
    def _emit_response_event(self, response: LLMResponse):
        """Emit an LLM response event.
        
        Args:
            response: LLM response
        """
        event_bus.publish(LLMResponseEvent(
            request_id=response.request_id,
            completion_id=response.completion_id,
            provider=response.provider,
            model=response.model,
            is_success=response.is_success,
            error=response.error,
            latency_ms=response.latency_ms,
            token_usage=response.usage,
            timestamp=datetime.now().isoformat()
        ))

class OpenAIBridge(LLMBridge):
    """Bridge for OpenAI models."""
    
    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None):
        """Initialize with model and API key.
        
        Args:
            model: Model name
            api_key: OpenAI API key (default: from environment)
        """
        super().__init__(model, ModelProvider.OPENAI)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        
        if not self.api_key:
            logger.warning("No OpenAI API key provided")
        
        try:
            # Import required libraries
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
            logger.debug(f"Initialized OpenAI client with model {model}")
        except ImportError:
            logger.error("Failed to import OpenAI library")
            self.client = None
    
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate text using OpenAI models.
        
        Args:
            request: LLM request
            
        Returns:
            LLM response
        """
        if not self.client:
            return LLMResponse(
                text="",
                request_id=request.request_id,
                provider=self.provider.value,
                model=self.model,
                error="OpenAI client not initialized"
            )
        
        try:
            # Emit request event
            self._emit_request_event(request)
            
            start_time = time.time()
            
            # Prepare messages
            messages = []
            
            # Add system message if provided
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            
            # Add provided messages if any
            if request.messages:
                messages.extend(request.messages)
            # Otherwise, add the prompt as a user message
            elif request.prompt:
                messages.append({"role": "user", "content": request.prompt})
            
            # Make API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p,
                presence_penalty=request.presence_penalty,
                frequency_penalty=request.frequency_penalty,
                stop=request.stop_sequences if request.stop_sequences else None,
                user=request.user_id
            )
            
            # Calculate latency
            latency_ms = int((time.time() - start_time) * 1000)
            
            # Extract response
            text = response.choices[0].message.content
            
            # Create LLM response
            llm_response = LLMResponse(
                text=text,
                request_id=request.request_id,
                completion_id=response.id,
                provider=self.provider.value,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                raw_response=response,
                finish_reason=response.choices[0].finish_reason,
                latency_ms=latency_ms
            )
            
            # Emit response event
            self._emit_response_event(llm_response)
            
            return llm_response
            
        except Exception as e:
            logger.error(f"Error generating with OpenAI: {str(e)}")
            
            # Create error response
            error_response = LLMResponse(
                text="",
                request_id=request.request_id,
                provider=self.provider.value,
                model=self.model,
                error=str(e),
                latency_ms=int((time.time() - start_time) * 1000) if 'start_time' in locals() else 0
            )
            
            # Emit response event
            self._emit_response_event(error_response)
            
            return error_response
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI models.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not self.client:
            logger.error("OpenAI client not initialized")
            return [[0.0] for _ in texts]  # Return dummy embeddings
        
        try:
            # Use text-embedding-3-small by default for embeddings
            embedding_model = "text-embedding-3-small"
            
            # Make API call
            response = self.client.embeddings.create(
                model=embedding_model,
                input=texts
            )
            
            # Extract embeddings
            embeddings = [data.embedding for data in response.data]
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings with OpenAI: {str(e)}")
            # Return dummy embeddings as fallback
            return [[0.0] for _ in texts]

class AnthropicBridge(LLMBridge):
    """Bridge for Anthropic models."""
    
    def __init__(self, model: str = "claude-3-opus-20240229", api_key: Optional[str] = None):
        """Initialize with model and API key.
        
        Args:
            model: Model name
            api_key: Anthropic API key (default: from environment)
        """
        super().__init__(model, ModelProvider.ANTHROPIC)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        
        if not self.api_key:
            logger.warning("No Anthropic API key provided")
        
        try:
            # Import required libraries
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
            logger.debug(f"Initialized Anthropic client with model {model}")
        except ImportError:
            logger.error("Failed to import Anthropic library")
            self.client = None
    
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate text using Anthropic models.
        
        Args:
            request: LLM request
            
        Returns:
            LLM response
        """
        if not self.client:
            return LLMResponse(
                text="",
                request_id=request.request_id,
                provider=self.provider.value,
                model=self.model,
                error="Anthropic client not initialized"
            )
        
        try:
            # Emit request event
            self._emit_request_event(request)
            
            start_time = time.time()
            
            # Prepare system prompt and messages
            system_prompt = request.system_prompt or ""
            
            # Convert messages to Anthropic format
            message_content = []
            
            if request.messages:
                # Extract messages, but exclude system messages
                for message in request.messages:
                    if message.get("role") != "system":
                        content = message.get("content", "")
                        if content:
                            message_content.append({
                                "type": "text",
                                "text": content
                            })
            # Otherwise, use the prompt directly
            elif request.prompt:
                message_content.append({
                    "type": "text",
                    "text": request.prompt
                })
            
            # Make API call
            response = self.client.messages.create(
                model=self.model,
                system=system_prompt,
                messages=[{
                    "role": "user",
                    "content": message_content
                }],
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p,
                stop_sequences=request.stop_sequences if request.stop_sequences else None
            )
            
            # Calculate latency
            latency_ms = int((time.time() - start_time) * 1000)
            
            # Extract response
            text = response.content[0].text
            
            # Create LLM response
            llm_response = LLMResponse(
                text=text,
                request_id=request.request_id,
                completion_id=response.id,
                provider=self.provider.value,
                model=response.model,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                },
                raw_response=response,
                latency_ms=latency_ms
            )
            
            # Emit response event
            self._emit_response_event(llm_response)
            
            return llm_response
            
        except Exception as e:
            logger.error(f"Error generating with Anthropic: {str(e)}")
            
            # Create error response
            error_response = LLMResponse(
                text="",
                request_id=request.request_id,
                provider=self.provider.value,
                model=self.model,
                error=str(e),
                latency_ms=int((time.time() - start_time) * 1000) if 'start_time' in locals() else 0
            )
            
            # Emit response event
            self._emit_response_event(error_response)
            
            return error_response
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings (not directly supported by Anthropic).
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors (falls back to OpenAI)
        """
        logger.warning("Embedding not directly supported by Anthropic, falling back to OpenAI")
        try:
            # Create OpenAI bridge for embeddings
            openai_bridge = OpenAIBridge()
            return openai_bridge.embed(texts)
        except Exception as e:
            logger.error(f"Error generating embeddings with fallback: {str(e)}")
            # Return dummy embeddings as fallback
            return [[0.0] for _ in texts]

class LlamaBridge(LLMBridge):
    """Bridge for Llama models using llama-cpp-python."""
    
    def __init__(self, model_path: str):
        """Initialize with model path.
        
        Args:
            model_path: Path to the model file
        """
        super().__init__(model_path, ModelProvider.LLAMA)
        self.model_path = model_path
        
        try:
            # Import required libraries
            from llama_cpp import Llama
            self.llm = Llama(model_path=model_path)
            logger.debug(f"Initialized Llama model from {model_path}")
        except ImportError:
            logger.error("Failed to import llama-cpp-python library")
            self.llm = None
    
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate text using Llama models.
        
        Args:
            request: LLM request
            
        Returns:
            LLM response
        """
        if not self.llm:
            return LLMResponse(
                text="",
                request_id=request.request_id,
                provider=self.provider.value,
                model=self.model_path,
                error="Llama model not initialized"
            )
        
        try:
            # Emit request event
            self._emit_request_event(request)
            
            start_time = time.time()
            
            # Prepare prompt
            prompt = request.prompt
            
            # Add system prompt if provided
            if request.system_prompt:
                prompt = f"{request.system_prompt}\n\n{prompt}"
            
            # Make API call
            response = self.llm(
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                stop=request.stop_sequences if request.stop_sequences else None
            )
            
            # Calculate latency
            latency_ms = int((time.time() - start_time) * 1000)
            
            # Extract response
            text = response["choices"][0]["text"]
            
            # Create LLM response
            llm_response = LLMResponse(
                text=text,
                request_id=request.request_id,
                provider=self.provider.value,
                model=self.model_path,
                usage={
                    "prompt_tokens": response["usage"]["prompt_tokens"],
                    "completion_tokens": response["usage"]["completion_tokens"],
                    "total_tokens": response["usage"]["total_tokens"]
                },
                raw_response=response,
                finish_reason=response["choices"][0].get("finish_reason"),
                latency_ms=latency_ms
            )
            
            # Emit response event
            self._emit_response_event(llm_response)
            
            return llm_response
            
        except Exception as e:
            logger.error(f"Error generating with Llama: {str(e)}")
            
            # Create error response
            error_response = LLMResponse(
                text="",
                request_id=request.request_id,
                provider=self.provider.value,
                model=self.model_path,
                error=str(e),
                latency_ms=int((time.time() - start_time) * 1000) if 'start_time' in locals() else 0
            )
            
            # Emit response event
            self._emit_response_event(error_response)
            
            return error_response
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Llama models.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not self.llm:
            logger.error("Llama model not initialized")
            return [[0.0] for _ in texts]  # Return dummy embeddings
        
        try:
            # Check if embedding is supported
            if hasattr(self.llm, "embed"):
                embeddings = []
                for text in texts:
                    embedding = self.llm.embed(text)
                    embeddings.append(embedding)
                return embeddings
            else:
                logger.warning("Embedding not supported by this Llama model, falling back to OpenAI")
                # Create OpenAI bridge for embeddings
                openai_bridge = OpenAIBridge()
                return openai_bridge.embed(texts)
                
        except Exception as e:
            logger.error(f"Error generating embeddings with Llama: {str(e)}")
            # Return dummy embeddings as fallback
            return [[0.0] for _ in texts]


class AIAdapter:
    """Adapter for different LLM bridges."""
    
    def __init__(self, default_bridge: Optional[LLMBridge] = None):
        """Initialize with default bridge.
        
        Args:
            default_bridge: Default LLM bridge
        """
        self.bridges: Dict[ModelProvider, LLMBridge] = {}
        self.default_bridge = default_bridge
        
        if default_bridge:
            self.register_bridge(default_bridge)
            
        logger.debug("Initialized AI Adapter")
    
    def register_bridge(self, bridge: LLMBridge) -> None:
        """Register an LLM bridge.
        
        Args:
            bridge: LLM bridge to register
        """
        self.bridges[bridge.provider] = bridge
        
        # Set as default if no default bridge
        if not self.default_bridge:
            self.default_bridge = bridge
            
        logger.debug(f"Registered {bridge.provider.value} bridge")
    
    def get_bridge(self, provider: ModelProvider) -> Optional[LLMBridge]:
        """Get a bridge by provider.
        
        Args:
            provider: Model provider
            
        Returns:
            LLM bridge or None if not found
        """
        return self.bridges.get(provider)
    
    def generate(self, request: Union[LLMRequest, str], provider: Optional[ModelProvider] = None, **kwargs) -> LLMResponse:
        """Generate text using an LLM bridge.
        
        Args:
            request: LLM request or prompt string
            provider: Optional provider to use
            **kwargs: Additional parameters for the request
            
        Returns:
            LLM response
        """
        # Get bridge
        bridge = self.bridges.get(provider) if provider else self.default_bridge
        
        if not bridge:
            return LLMResponse(
                text="",
                request_id=str(uuid.uuid4()),
                error="No LLM bridge available"
            )
        
        # Convert string to request if needed
        if isinstance(request, str):
            request = LLMRequest(prompt=request, **kwargs)
        elif kwargs:
            # Update request with kwargs
            for key, value in kwargs.items():
                if hasattr(request, key):
                    setattr(request, key, value)
        
        # Generate response
        return bridge.generate(request)
    
    def chat(self, messages: List[Dict[str, str]], provider: Optional[ModelProvider] = None, **kwargs) -> LLMResponse:
        """Generate a response in a chat context.
        
        Args:
            messages: List of message dictionaries with role and content
            provider: Optional provider to use
            **kwargs: Additional parameters
            
        Returns:
            LLM response
        """
        # Get bridge
        bridge = self.bridges.get(provider) if provider else self.default_bridge
        
        if not bridge:
            return LLMResponse(
                text="",
                request_id=str(uuid.uuid4()),
                error="No LLM bridge available"
            )
        
        # Generate response
        return bridge.chat(messages, **kwargs)
    
    def embed(self, texts: List[str], provider: Optional[ModelProvider] = None) -> List[List[float]]:
        """Generate embeddings for texts.
        
        Args:
            texts: List of texts to embed
            provider: Optional provider to use
            
        Returns:
            List of embedding vectors
        """
        # Get bridge
        bridge = self.bridges.get(provider) if provider else self.default_bridge
        
        if not bridge:
            logger.error("No LLM bridge available for embeddings")
            return [[0.0] for _ in texts]  # Return dummy embeddings
        
        # Generate embeddings
        return bridge.embed(texts)


# Create a default instance with OpenAI
try:
    default_bridge = OpenAIBridge()
    ai_adapter = AIAdapter(default_bridge)
except Exception as e:
    logger.error(f"Error creating default AI adapter: {str(e)}")
    ai_adapter = AIAdapter()
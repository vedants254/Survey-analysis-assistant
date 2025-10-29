"""
LLM Provider Abstraction Layer
Unified interface for multiple LLM providers: OpenAI, Anthropic, Ollama, Google Generative AI
Default provider: Google Gemini
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, AsyncIterator
from pydantic import BaseModel
import httpx
import json
from dataclasses import dataclass
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"  
    OLLAMA = "ollama"
    GEMINI = "gemini"
    GROQ = "groq"

@dataclass
class LLMMessage:
    """Standardized message format across all providers"""
    role: str  # system, user, assistant
    content: str
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class LLMResponse:
    """Standardized response format across all providers"""
    content: str
    provider: str
    model: str
    tokens_used: Optional[int] = None
    finish_reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class BaseLLMProvider(ABC):
    """Abstract base class for all LLM providers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = config.get("model")
        self.max_tokens = config.get("max_tokens", 4096)
        self.temperature = config.get("temperature", 0.1)
    
    @abstractmethod
    async def generate(
        self, 
        messages: List[LLMMessage], 
        **kwargs
    ) -> LLMResponse:
        """Generate a response from the LLM"""
        pass
    
    @abstractmethod
    async def generate_stream(
        self, 
        messages: List[LLMMessage], 
        **kwargs
    ) -> AsyncIterator[str]:
        """Generate a streaming response from the LLM"""
        pass
    
    @abstractmethod
    def validate_config(self) -> bool:
        """Validate provider configuration"""
        pass

class GeminiProvider(BaseLLMProvider):
    """Google Gemini provider implementation (Default)"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key") or os.getenv("GOOGLE_API_KEY")
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self.model = config.get("model", "gemini-1.5-flash")
        
        if not self.api_key:
            logger.error("❌ GOOGLE_API_KEY not found! Set environment variable: export GOOGLE_API_KEY='your_key_here'")
            raise ValueError("Google API key is required for LLM functionality")
    
    def validate_config(self) -> bool:
        return bool(self.api_key)
    
    async def generate(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        """Generate response using Google Generative AI API"""
        try:
            
            # Convert messages to Google format
            contents = []
            system_instruction = None
            
            for msg in messages:
                if msg.role == "system":
                    system_instruction = msg.content
                elif msg.role == "user":
                    contents.append({
                        "parts": [{"text": msg.content}],
                        "role": "user"
                    })
                elif msg.role == "assistant":
                    contents.append({
                        "parts": [{"text": msg.content}],
                        "role": "model"
                    })
            
            payload = {
                "contents": contents,
                "generationConfig": {
                    "maxOutputTokens": kwargs.get("max_tokens", self.max_tokens),
                    "temperature": kwargs.get("temperature", self.temperature),
                }
            }
            
            if system_instruction:
                payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}",
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status()
                data = response.json()
                
                content = ""
                if "candidates" in data and data["candidates"]:
                    candidate = data["candidates"][0]
                    if "content" in candidate and "parts" in candidate["content"]:
                        content = candidate["content"]["parts"][0]["text"]
                
                return LLMResponse(
                    content=content,
                    provider="gemini",
                    model=self.model,
                    tokens_used=data.get("usageMetadata", {}).get("totalTokenCount"),
                    finish_reason=data["candidates"][0].get("finishReason") if data.get("candidates") else None,
                    metadata={"usage": data.get("usageMetadata")}
                )
                
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise Exception(f"Google Gemini API failed: {e}. Please check your GOOGLE_API_KEY and network connection.")
    
    async def generate_stream(self, messages: List[LLMMessage], **kwargs) -> AsyncIterator[str]:
        """Stream response from Gemini"""
        response = await self.generate(messages, **kwargs)
        # Simulate streaming by yielding in chunks
        words = response.content.split()
        for i in range(0, len(words), 3):
            yield " ".join(words[i:i+3]) + " "
    

class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT provider implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key") or os.getenv("OPENAI_API_KEY")
        self.base_url = "https://api.openai.com/v1"
        self.model = config.get("model", "gpt-4")
    
    def validate_config(self) -> bool:
        return bool(self.api_key)
    
    async def generate(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        """Generate response using OpenAI API"""
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        
        try:
            openai_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]
            
            payload = {
                "model": self.model,
                "messages": openai_messages,
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "temperature": kwargs.get("temperature", self.temperature),
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=30.0
                )
                response.raise_for_status()
                data = response.json()
                
                return LLMResponse(
                    content=data["choices"][0]["message"]["content"],
                    provider="openai",
                    model=self.model,
                    tokens_used=data.get("usage", {}).get("total_tokens"),
                    finish_reason=data["choices"][0].get("finish_reason"),
                    metadata={"usage": data.get("usage")}
                )
                
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise Exception(f"OpenAI API failed: {e}. Please check your OPENAI_API_KEY and network connection.")
    
    async def generate_stream(self, messages: List[LLMMessage], **kwargs) -> AsyncIterator[str]:
        """Stream response from OpenAI"""
        response = await self.generate(messages, **kwargs)
        words = response.content.split()
        for i in range(0, len(words), 2):
            yield " ".join(words[i:i+2]) + " "
    

class OllamaProvider(BaseLLMProvider):
    """Ollama local/open-source model provider"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get("base_url", "http://localhost:11434")
        self.model = config.get("model", "llama2:13b")
        self.timeout = config.get("timeout", 300)
    
    def validate_config(self) -> bool:
        return bool(self.base_url and self.model)
    
    async def generate(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        """Generate response using Ollama API"""
        try:
            prompt = self._messages_to_prompt(messages)
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", self.temperature),
                    "num_predict": kwargs.get("max_tokens", self.max_tokens),
                }
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()
                
                return LLMResponse(
                    content=data["response"],
                    provider="ollama",
                    model=self.model,
                    tokens_used=None,
                    finish_reason="stop",
                    metadata={
                        "eval_count": data.get("eval_count"),
                        "eval_duration": data.get("eval_duration")
                    }
                )
                
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise Exception(f"Ollama API failed: {e}. Please check if Ollama is running on {self.base_url}")
    
    async def generate_stream(self, messages: List[LLMMessage], **kwargs) -> AsyncIterator[str]:
        """Stream response from Ollama"""
        response = await self.generate(messages, **kwargs)
        words = response.content.split()
        for word in words:
            yield word + " "
    
    def _messages_to_prompt(self, messages: List[LLMMessage]) -> str:
        """Convert messages to a single prompt string for Ollama"""
        prompt_parts = []
        for msg in messages:
            if msg.role == "system":
                prompt_parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                prompt_parts.append(f"Human: {msg.content}")
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}")
        
        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)
    

class GroqProvider(BaseLLMProvider):
    """Groq fast inference provider implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key") or os.getenv("GROQ_API_KEY")
        self.base_url = "https://api.groq.com/openai/v1"
        self.model = config.get("model", "mixtral-8x7b-32768")
        
        if not self.api_key:
            logger.error("❌ GROQ_API_KEY not found! Set environment variable: export GROQ_API_KEY='your_key_here'")
            raise ValueError("Groq API key is required for LLM functionality")
    
    def validate_config(self) -> bool:
        return bool(self.api_key)
    
    async def generate(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        """Generate response using Groq API (OpenAI-compatible)"""
        if not self.api_key:
            raise ValueError("Groq API key is required. Set GROQ_API_KEY environment variable.")
        
        try:
            groq_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]
            
            payload = {
                "model": self.model,
                "messages": groq_messages,
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "temperature": kwargs.get("temperature", self.temperature),
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=30.0
                )
                response.raise_for_status()
                data = response.json()
                
                return LLMResponse(
                    content=data["choices"][0]["message"]["content"],
                    provider="groq",
                    model=self.model,
                    tokens_used=data.get("usage", {}).get("total_tokens"),
                    finish_reason=data["choices"][0].get("finish_reason"),
                    metadata={"usage": data.get("usage")}
                )
                
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            raise Exception(f"Groq API failed: {e}. Please check your GROQ_API_KEY and network connection.")
    
    async def generate_stream(self, messages: List[LLMMessage], **kwargs) -> AsyncIterator[str]:
        """Stream response from Groq"""
        response = await self.generate(messages, **kwargs)
        words = response.content.split()
        for i in range(0, len(words), 2):
            yield " ".join(words[i:i+2]) + " "
    

class LLMProviderManager:
    """Manager class for handling multiple LLM providers"""
    
    def __init__(self):
        self.providers: Dict[str, BaseLLMProvider] = {}
        self.default_provider = None
    
    def register_provider(self, name: str, provider: BaseLLMProvider, is_default: bool = False):
        """Register an LLM provider"""
        if provider.validate_config():
            self.providers[name] = provider
            if is_default or not self.default_provider:
                self.default_provider = name
            logger.info(f"Registered LLM provider: {name}")
        else:
            logger.warning(f"Failed to register LLM provider {name}: Invalid configuration")
    
    def get_provider(self, name: Optional[str] = None) -> BaseLLMProvider:
        """Get a provider by name or return default"""
        provider_name = name or self.default_provider
        if provider_name not in self.providers:
            raise ValueError(f"Provider {provider_name} not found")
        return self.providers[provider_name]
    
    async def generate(
        self, 
        messages: List[LLMMessage], 
        provider: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using specified or default provider"""
        llm_provider = self.get_provider(provider)
        return await llm_provider.generate(messages, **kwargs)
    
    async def generate_stream(
        self, 
        messages: List[LLMMessage], 
        provider: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Generate streaming response using specified or default provider"""
        llm_provider = self.get_provider(provider)
        async for chunk in llm_provider.generate_stream(messages, **kwargs):
            yield chunk
    
    def list_providers(self) -> List[str]:
        """List all registered providers"""
        return list(self.providers.keys())

def create_llm_manager() -> LLMProviderManager:
    """Factory function to create and configure LLM provider manager"""
    manager = LLMProviderManager()
    
    # Load configuration from environment
    default_provider = os.getenv("DEFAULT_LLM_PROVIDER", "gemini")
    
    # Register Gemini as primary provider
    try:
        gemini_config = {
            "api_key": os.getenv("GOOGLE_API_KEY"),
            "model": os.getenv("GOOGLE_MODEL", "gemini-1.5-flash"),
            "max_tokens": int(os.getenv("GOOGLE_MAX_TOKENS", "4096")),
            "temperature": float(os.getenv("GOOGLE_TEMPERATURE", "0.1"))
        }
        manager.register_provider(
            "gemini", 
            GeminiProvider(gemini_config),
            is_default=(default_provider == "gemini")
        )
    except Exception as e:
        logger.error(f"Failed to register Gemini provider: {e}")
    
    # Register OpenAI if configured
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            openai_config = {
                "api_key": openai_key,
                "model": os.getenv("OPENAI_MODEL", "gpt-4"),
                "max_tokens": int(os.getenv("OPENAI_MAX_TOKENS", "4096")),
                "temperature": float(os.getenv("OPENAI_TEMPERATURE", "0.1"))
            }
            manager.register_provider(
                "openai", 
                OpenAIProvider(openai_config),
                is_default=(default_provider == "openai")
            )
        except Exception as e:
            logger.error(f"Failed to register OpenAI provider: {e}")
    
    # Register Groq if configured
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        try:
            groq_config = {
                "api_key": groq_key,
                "model": os.getenv("GROQ_MODEL", "mixtral-8x7b-32768"),
                "max_tokens": int(os.getenv("GROQ_MAX_TOKENS", "4096")),
                "temperature": float(os.getenv("GROQ_TEMPERATURE", "0.1"))
            }
            manager.register_provider(
                "groq", 
                GroqProvider(groq_config),
                is_default=(default_provider == "groq")
            )
        except Exception as e:
            logger.error(f"Failed to register Groq provider: {e}")
    
    # Register Ollama (always available for local development)
    try:
        ollama_config = {
            "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            "model": os.getenv("OLLAMA_MODEL", "llama2:13b"),
            "max_tokens": int(os.getenv("OLLAMA_MAX_TOKENS", "4096")),
            "temperature": float(os.getenv("OLLAMA_TEMPERATURE", "0.1")),
            "timeout": int(os.getenv("OLLAMA_TIMEOUT", "300"))
        }
        manager.register_provider(
            "ollama", 
            OllamaProvider(ollama_config),
            is_default=(default_provider == "ollama")
        )
    except Exception as e:
        logger.error(f"Failed to register Ollama provider: {e}")
    
    # Ensure we have at least one provider
    if not manager.providers:
        logger.error("❌ NO LLM PROVIDERS AVAILABLE! Please configure at least one API key:")
        logger.error("- Google Gemini: export GOOGLE_API_KEY='your_key_here'")
        logger.error("- OpenAI: export OPENAI_API_KEY='your_key_here'")
        logger.error("- Groq: export GROQ_API_KEY='your_key_here'")
        logger.error("- Ollama: Start local Ollama server on http://localhost:11434")
        raise RuntimeError("No LLM providers configured. Cannot perform AI analysis without API keys.")
    
    logger.info(f"LLM Manager initialized with providers: {manager.list_providers()}")
    logger.info(f"Default provider: {manager.default_provider}")
    
    return manager

# Global instance
llm_manager = create_llm_manager()
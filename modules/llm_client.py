"""
Universal LLM Client
Supports: Groq (FREE, primary) -> HuggingFace (FREE, fallback)
No OpenAI. No local models. Works anywhere with internet.
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass

from groq import Groq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.callbacks.manager import CallbackManagerForLLMRun

from config import GROQ_API_KEY, GROQ_MODEL


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""
    content: str
    model: str
    provider: str
    usage: Dict = None
    raw_response: any = None


class GroqClient:
    """
    Groq API client - FREE tier available
    Sign up: https://console.groq.com (no credit card, instant access)
    Free limits: 20 requests/minute, 1,500,000 tokens/day
    """
    
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or GROQ_API_KEY
        self.model = model or GROQ_MODEL
        
        if not self.api_key:
            raise ValueError(
                "Groq API key required. Get free key at https://console.groq.com/keys"
            )
        
        self.client = Groq(api_key=self.api_key)
        print(f"🔥 Groq client initialized (model: {self.model})")
    
    def chat(self, messages: List[Dict], temperature: float = 0.0, 
             max_tokens: int = 1024, json_mode: bool = False) -> LLMResponse:
        """
        Send chat completion request to Groq.
        
        Args:
            messages: List of {"role": "system|user|assistant", "content": "..."}
            temperature: 0.0 = deterministic, 1.0 = creative
            max_tokens: Maximum response length
            json_mode: Force JSON output (for classification)
            
        Returns:
            LLMResponse with standardized fields
        """
        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            
            response = self.client.chat.completions.create(**kwargs)
            
            return LLMResponse(
                content=response.choices[0].message.content,
                model=self.model,
                provider="groq",
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                raw_response=response
            )
            
        except Exception as e:
            print(f"❌ Groq API error: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if Groq API is accessible."""
        try:
            self.client.models.list()
            return True
        except:
            return False


class UnifiedLLM(BaseChatModel):
    """
    LangChain-compatible LLM that auto-selects free cloud provider.
    Priority: Groq -> HuggingFace -> Error with helpful message
    """
    
    temperature: float = 0.0
    max_tokens: int = 1024
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = None
        self._provider = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Auto-detect and initialize the best available free LLM."""
        providers = []
        
        # Try Groq first (best free option)
        if GROQ_API_KEY:
            try:
                client = GroqClient()
                if client.is_available():
                    self._client = client
                    self._provider = "groq"
                    return
            except Exception as e:
                providers.append(f"Groq: {e}")
        
        # No provider available
        error_msg = (
            "\n❌ NO FREE LLM PROVIDER AVAILABLE!\n"
            "\nQuick fix (choose one):\n"
            "1. GROQ (Recommended - fastest, most reliable):\n"
            "   → Go to https://console.groq.com/keys\n"
            "   → Sign up with Google/GitHub (30 seconds, no credit card)\n"
            "   → Create API key\n"
            "   → Add to .env: GROQ_API_KEY=gsk_your_key_here\n"
            "\n"
            f"Errors encountered:\n" + "\n".join(f"  - {p}" for p in providers)
        )
        raise RuntimeError(error_msg)
    
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """LangChain _generate implementation."""
        # Convert LangChain messages to dict format
        msg_dicts = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                msg_dicts.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                msg_dicts.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                msg_dicts.append({"role": "assistant", "content": msg.content})
        
        # Check if JSON mode requested
        json_mode = kwargs.get("response_format") == {"type": "json_object"}
        
        response = self._client.chat(
            messages=msg_dicts,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            json_mode=json_mode
        )
        
        # Create LangChain ChatGeneration
        message = AIMessage(content=response.content)
        generation = ChatGeneration(message=message)
        
        return ChatResult(generations=[generation])
    
    @property
    def _llm_type(self) -> str:
        return f"unified_{self._provider}"
    
    @property
    def _identifying_params(self) -> Dict:
        return {
            "provider": self._provider,
            "model": self._client.model if self._client else None,
            "temperature": self.temperature
        }


# ─── Convenience Functions ─────────────────────────────────────────
def get_llm(temperature: float = 0.0, max_tokens: int = 1024):
    """
    Get a ready-to-use LLM instance.
    
    Returns:
        UnifiedLLM instance (Groq or HuggingFace)
    """
    return UnifiedLLM(temperature=temperature, max_tokens=max_tokens)


def check_setup() -> Dict:
    """
    Check which free LLM providers are available.
    
    Returns:
        Dict with availability status and instructions
    """
    status = {
        "groq": {"available": False, "key_set": bool(GROQ_API_KEY)},
        "recommendation": ""
    }
    
    if GROQ_API_KEY:
        try:
            client = GroqClient()
            status["groq"]["available"] = client.is_available()
        except:
            pass
    
    # Recommendation
    if status["groq"]["available"]:
        status["recommendation"] = "✅ Using Groq (fastest, recommended)"
    else:
        status["recommendation"] = "❌ No provider available. See setup instructions."
    
    return status


# ─── Standalone test ───────────────────────────────────────────────
if __name__ == "__main__":
    print("🔍 Checking free LLM providers...\n")
    
    status = check_setup()
    for provider, info in status.items():
        if provider != "recommendation":
            print(f"{provider.upper()}:")
            print(f"  Key set: {'✅' if info['key_set'] else '❌'}")
            print(f"  Available: {'✅' if info['available'] else '❌'}")
            print()
    
    print(status["recommendation"])
    
    # Test generation
    if status["groq"]["available"]:
        print("\n🧪 Testing LLM generation...")
        llm = get_llm()
        
        from langchain_core.prompts import ChatPromptTemplate
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("human", "Say 'RAG system is working!' and nothing else.")
        ])
        
        result = prompt | llm
        response = result.invoke({})
        print(f"Response: {response.content}")
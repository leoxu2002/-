import os
from typing import Any, List, Optional, Dict

from src.core.settings import Settings
from src.libs.llm.base_llm import BaseLLM, ChatResponse, Message

class QwenLLMError(RuntimeError):
    """Raised when Qwen (DashScope) API call fails."""
    
    pass

"""
Qwen LLM

"""
class QwenLLM(BaseLLM):
    """
    Qwen LLM
    """
    DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com"

    SUPPORTED_MODELS = {
        "qwen-turbo": "Qwen Turbo - 快速响应",
        "qwen-plus": "Qwen Plus - 均衡性能",
        "qwen-max": "Qwen Max - 最高性能",
        "qwen-max-longcontext": "Qwen Max (长上下文)",
    }

    def __init__(self, 
                 settings: Settings, 
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 **kwargs: Any,
                 )->None:
        """
        Initialize the Qwen LLM.
        """
        self.model = settings.llm.model
        self.default_temperature = settings.llm.temperature
        self.default_max_tokens = settings.llm.max_tokens
        self.api_key = (
            api_key 
            or getattr(settings.llm, 'api_key', None)
            or os.environ.get("DASHSCOPE_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "Qwen (DashScope) API key not provided. Set in settings.yaml (llm.api_key), "
                "DASHSCOPE_API_KEY environment variable, or pass api_key parameter."
            )
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self._extra_config = kwargs

    def chat(self, 
             messages: List[Message], 
             trace: Any | None = None, 
             **kwargs: Any) -> ChatResponse:
        self.validate_messages(messages)
        temperature = kwargs.get("temperature", self.default_temperature)
        max_tokens = kwargs.get("max_tokens", self.default_max_tokens)
        model = kwargs.get("model", self.model)
        api_messages = [{"role": m.role, "content": m.content} for m in messages]

        try:
            response_data = self._call_api(
                messages=api_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                model=model,
            )
            content = response_data["choices"][0]["message"]["content"]
            usage = response_data["usage"]
            return ChatResponse(
                content=content,
                model=response_data.get("model", model),
                usage=usage,
                raw_response=response_data,
            )
        
        except KeyError as e:
            raise QwenLLMError(
                f"[Qwen] Unexpected response format: missing key {e}"
            ) from e
        except Exception as e:
            if isinstance(e, QwenLLMError):
                raise
            raise QwenLLMError(
                f"[Qwen] API call failed: {type(e).__name__}: {e}"
            ) from e

    def _call_api(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        """Make the actual API call to DashScope.
        
        Args:
            messages: Messages in API format.
            model: Model identifier.
            temperature: Generation temperature.
            max_tokens: Maximum tokens to generate.
        
        Returns:
            Raw API response as dictionary.
        
        Raises:
            QwenLLMError: If the API call fails.
        """
        import httpx
        
        # DashScope API 端点
        url = f"{self.base_url.rstrip('/')}/api/v1/services/aigc/text-generation/generation"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        # DashScope 请求格式
        payload = {
            "model": model,
            "input": {
                "messages": messages
            },
            "parameters": {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "result_format": "message",  # 返回消息格式
            },
        }
        
        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, json=payload, headers=headers)
                
                if response.status_code != 200:
                    error_detail = self._parse_error_response(response)
                    raise QwenLLMError(
                        f"[Qwen] API error (HTTP {response.status_code}): {error_detail}"
                    )
                
                return response.json()
                
        except httpx.TimeoutException as e:
            raise QwenLLMError(
                f"[Qwen] Request timed out after 60 seconds"
            ) from e
        except httpx.RequestError as e:
            raise QwenLLMError(
                f"[Qwen] Connection failed: {type(e).__name__}: {e}"
            ) from e
    
    def _parse_error_response(self, response: Any) -> str:
        """Parse error details from API response.
        
        Args:
            response: The HTTP response object.
        
        Returns:
            Human-readable error message.
        """
        try:
            error_data = response.json()
            # DashScope 错误格式
            if "code" in error_data:
                return f"{error_data.get('code', '')}: {error_data.get('message', '')}"
            if "error" in error_data:
                return str(error_data["error"])
            return response.text
        except Exception:
            return response.text or "Unknown error"
        return super().chat(messages, trace, **kwargs)




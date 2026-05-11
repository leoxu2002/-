"""Qwen Vision LLM implementation for DashScope API.

This module provides Qwen Vision LLM implementation for multimodal
interactions (text + image). Supports Qwen-VL-Max and Qwen-VL-Plus models
for image understanding tasks like image captioning, visual question answering,
and document analysis.

DashScope API Documentation:
https://help.aliyun.com/document_detail/2712195.html
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Any, Optional

from src.libs.llm.base_llm import ChatResponse, Message
from src.libs.llm.base_vision_llm import BaseVisionLLM, ImageInput


class QwenVisionLLMError(RuntimeError):
    """Raised when Qwen Vision (DashScope) API call fails."""
    pass


class QwenVisionLLM(BaseVisionLLM):
    """Qwen Vision LLM provider implementation for DashScope API.
    
    This class implements the BaseVisionLLM interface using the DashScope
    API for Qwen-VL models. Supports both Qwen-VL-Max and Qwen-VL-Plus.
    
    Supported Models:
        - qwen-vl-max: Best performance, recommended for complex tasks
        - qwen-vl-plus: Faster, lower cost, good for batch processing
        - qwen-vl-ocr: Optimized for OCR and document understanding
    
    Attributes:
        api_key: The DashScope API key for authentication.
        base_url: The DashScope API base URL.
        model: The model identifier (e.g., 'qwen-vl-max').
        max_image_size: Maximum image dimension in pixels (default 2048).
        default_temperature: Default temperature for generation.
        default_max_tokens: Default max tokens for generation.
    
    Example:
        >>> from src.core.settings import load_settings
        >>> settings = load_settings('config/settings.yaml')
        >>> vision_llm = QwenVisionLLM(settings)
        >>> image = ImageInput(path="diagram.png")
        >>> response = vision_llm.chat_with_image("Describe this", image)
    """
    
    DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com"
    DEFAULT_MAX_IMAGE_SIZE = 2048
    
    SUPPORTED_MODELS = {
        "qwen3-vl-plus": "Qwen3 VL Plus - 快速响应，性价比高",
        "qwen3-vl-flash": "Qwen3 VL Flash - 文档理解优化",
    }
    
    def __init__(
        self,
        settings: Any,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_image_size: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Qwen Vision LLM provider.
        
        Args:
            settings: Application settings containing vision_llm configuration.
            api_key: Optional API key override.
            base_url: Optional base URL override.
            max_image_size: Maximum image dimension in pixels for auto-compression.
            **kwargs: Additional configuration overrides.
        
        Raises:
            ValueError: If required configuration is missing.
        """
        vision_settings = getattr(settings, "vision_llm", None)
        
        self.default_temperature = getattr(settings.llm, 'temperature', 0.0)
        self.default_max_tokens = getattr(settings.llm, 'max_tokens', 4096)
        
        vision_model = getattr(vision_settings, 'model', None) if vision_settings else None
        self.model = vision_model or "qwen-vl-max"
        
        vision_max_size = getattr(vision_settings, 'max_image_size', None) if vision_settings else None
        self.max_image_size = max_image_size or vision_max_size or self.DEFAULT_MAX_IMAGE_SIZE
        
        self.api_key = api_key
        if not self.api_key and vision_settings:
            self.api_key = getattr(vision_settings, 'api_key', None)
        if not self.api_key:
            self.api_key = getattr(settings.llm, 'api_key', None)
        if not self.api_key:
            self.api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Qwen (DashScope) API key not provided. Set in settings.yaml (vision_llm.api_key), "
                "DASHSCOPE_API_KEY environment variable, or pass api_key parameter."
            )
        
        vision_base_url = getattr(vision_settings, 'base_url', None) if vision_settings else None
        self.base_url = base_url or vision_base_url or self.DEFAULT_BASE_URL
        
        self._extra_config = kwargs
    
    def chat_with_image(
        self,
        text: str,
        image: ImageInput,
        messages: Optional[list[Message]] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Generate a response based on text prompt and image input.
        
        Args:
            text: The text prompt or question about the image.
            image: The image input (path, bytes, or base64).
            messages: Optional conversation history for context.
            trace: Optional TraceContext for observability.
            **kwargs: Override parameters (temperature, max_tokens, etc.).
        
        Returns:
            ChatResponse containing the generated text and metadata.
        
        Raises:
            ValueError: If text or image input is invalid.
            QwenVisionLLMError: If API call fails.
        """
        self.validate_text(text)
        self.validate_image(image)
        
        processed_image = self.preprocess_image(
            image,
            max_size=(self.max_image_size, self.max_image_size)
        )
        
        image_base64 = self._get_image_base64(processed_image)
        
        temperature = kwargs.get("temperature", self.default_temperature)
        max_tokens = kwargs.get("max_tokens", self.default_max_tokens)
        model = kwargs.get("model", self.model)
        
        api_messages = []
        if messages:
            api_messages.extend([{"role": m.role, "content": m.content} for m in messages])
        
        current_message = {
            "role": "user",
            "content": [
                {"text": text},
                {
                    "image": f"data:{processed_image.mime_type};base64,{image_base64}"
                }
            ]
        }
        api_messages.append(current_message)
        
        try:
            response_data = self._call_api(
                messages=api_messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            content = response_data["output"]["choices"][0]["message"]["content"][0]["text"]
            usage = response_data.get("usage", {})
            
            return ChatResponse(
                content=content,
                model=response_data.get("model", model),
                usage=usage,
                raw_response=response_data,
            )
        
        except KeyError as e:
            raise QwenVisionLLMError(
                f"[Qwen Vision] Unexpected response format: missing key {e}"
            ) from e
        except Exception as e:
            if isinstance(e, QwenVisionLLMError):
                raise
            raise QwenVisionLLMError(
                f"[Qwen Vision] API call failed: {type(e).__name__}: {e}"
            ) from e
    
    def preprocess_image(
        self,
        image: ImageInput,
        max_size: Optional[tuple[int, int]] = None,
    ) -> ImageInput:
        """Preprocess image before sending to Vision API.
        
        Compresses image if it exceeds max_size to reduce payload size.
        
        Args:
            image: The input image to preprocess.
            max_size: Maximum dimensions (width, height) in pixels.
        
        Returns:
            Preprocessed ImageInput with compressed data if needed.
        """
        if not max_size:
            return image
        
        try:
            from PIL import Image
        except ImportError:
            return image
        
        if image.data:
            image_bytes = image.data
        elif image.path:
            image_bytes = Path(image.path).read_bytes()
        elif image.base64:
            return image
        else:
            return image
        
        img = Image.open(io.BytesIO(image_bytes))
        width, height = img.size
        
        max_width, max_height = max_size
        if width <= max_width and height <= max_height:
            return image
        
        ratio = min(max_width / width, max_height / height)
        new_size = (int(width * ratio), int(height * ratio))
        
        img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
        
        buffer = io.BytesIO()
        img_format = img.format or "PNG"
        img_resized.save(buffer, format=img_format)
        compressed_bytes = buffer.getvalue()
        
        return ImageInput(
            data=compressed_bytes,
            mime_type=image.mime_type
        )
    
    def _get_image_base64(self, image: ImageInput) -> str:
        """Convert ImageInput to base64 string."""
        try:
            if image.base64:
                return image.base64
            elif image.data:
                return base64.b64encode(image.data).decode("utf-8")
            elif image.path:
                image_bytes = Path(image.path).read_bytes()
                return base64.b64encode(image_bytes).decode("utf-8")
            else:
                raise ValueError("ImageInput has no valid data source")
        except Exception as e:
            raise QwenVisionLLMError(
                f"[Qwen Vision] Failed to encode image: {e}"
            ) from e
    
    def _call_api(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> dict:
        """Make HTTP request to the DashScope Vision API.
        
        Args:
            messages: List of API-formatted messages with image content.
            model: Model identifier.
            temperature: Generation temperature.
            max_tokens: Maximum tokens to generate.
        
        Returns:
            API response as dictionary.
        
        Raises:
            QwenVisionLLMError: If API call fails.
        """
        import httpx
        
        url = f"{self.base_url.rstrip('/')}/api/v1/services/aigc/multimodal-generation/generation"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": model,
            "input": {
                "messages": messages
            },
            "parameters": {
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        }
        
        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(url, json=payload, headers=headers)
                
                if response.status_code != 200:
                    error_detail = self._parse_error_response(response)
                    raise QwenVisionLLMError(
                        f"[Qwen Vision] API error (HTTP {response.status_code}): {error_detail}"
                    )
                
                return response.json()
                
        except httpx.TimeoutException as e:
            raise QwenVisionLLMError(
                "[Qwen Vision] Request timed out after 120 seconds"
            ) from e
        except httpx.RequestError as e:
            raise QwenVisionLLMError(
                f"[Qwen Vision] Connection failed: {type(e).__name__}: {e}"
            ) from e
    
    def _parse_error_response(self, response: Any) -> str:
        """Parse error details from API response."""
        try:
            error_data = response.json()
            if "code" in error_data:
                return f"{error_data.get('code', '')}: {error_data.get('message', '')}"
            if "error" in error_data:
                return str(error_data["error"])
            return response.text
        except Exception:
            return response.text or "Unknown error"

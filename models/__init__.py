from gemini_flash import GeminiFlash
from qwenvl_local import HuggingfaceQwenVL

MODEL_MAPPING = {
    "gemini-flash": GeminiFlash,
    "qwen2-vl": HuggingfaceQwenVL,
}

__all__ = [MODEL_MAPPING]
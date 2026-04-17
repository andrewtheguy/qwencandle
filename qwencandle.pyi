"""Python bindings for Qwen3-ASR inference via Candle.

Supports `Qwen/Qwen3-ASR-0.6B` (default) and `Qwen/Qwen3-ASR-1.7B`.

`model_id` accepts:
- a HuggingFace model ID like `"Qwen/Qwen3-ASR-0.6B"` or `"Qwen/Qwen3-ASR-1.7B"`
- a local safetensors model directory
- a local quantized GGUF directory containing `model.gguf`
- a direct local `.gguf` file path when the tokenizer files are alongside it
"""

import numpy as np
import numpy.typing as npt

DEFAULT_MODEL_ID: str
SUPPORTED_LANGUAGES: list[str]

def is_cuda_available() -> bool: ...
def is_metal_available() -> bool: ...

class QwenAsr:
    """Speech-to-text model loader for CPU, Metal, or CUDA."""

    def __init__(
        # `model_id` may be a HuggingFace model ID, a local model directory,
        # or a local GGUF path/directory produced by `qwencandle quantize`.
        self, device: str, model_id: str | None = None
    ) -> None: ...
    def transcribe(
        self,
        samples: npt.NDArray[np.float32],
        *,
        language: str | None = None,
        context: str | None = None,
    ) -> str: ...

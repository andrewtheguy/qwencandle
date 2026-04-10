import numpy as np
import numpy.typing as npt

DEFAULT_MODEL_ID: str
SUPPORTED_LANGUAGES: list[str]

def is_cuda_available() -> bool: ...
def is_metal_available() -> bool: ...

class QwenAsr:
    def __init__(
        self, device: str, model_id: str | None = None
    ) -> None: ...
    def transcribe(
        self,
        samples: npt.NDArray[np.float32],
        *,
        language: str | None = None,
        context: str | None = None,
    ) -> str: ...

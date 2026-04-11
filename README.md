# qwencandle

Qwen3-ASR-0.6B speech-to-text inference in Rust, built on [Candle](https://github.com/huggingface/candle).

## Motivation

The main existing Python implementations of Qwen ASR (via PyTorch / Transformers) suffer from memory leaks when running inference repeatedly on Apple Silicon with Metal/MPS. Memory grows with each transcription call and is never fully reclaimed, eventually forcing a restart. This makes them unsuitable for long-running services or batch processing workflows on macOS.

This project reimplements Qwen3-ASR inference in Rust using HuggingFace's [Candle](https://github.com/huggingface/candle) framework, which provides correct Metal GPU support without the memory leak issues. The result is a lightweight, memory-stable binary (and Python library) that can transcribe audio indefinitely on Metal without degradation.

## Build

```
cargo build --release
```

With Metal GPU + Accelerate BLAS (macOS):

```
cargo build --release --features metal,accelerate
```

With Intel MKL BLAS (Linux x86_64):

```
cargo build --release --features mkl
```

## Usage

Input is WAV float32 16kHz mono on stdin. Use ffmpeg to convert any audio format:

```
ffmpeg -i audio.mp3 -ac 1 -ar 16000 -f wav -acodec pcm_f32le - | ./target/release/qwencandle
```

The model is automatically downloaded from HuggingFace on first use and cached in `~/.cache/huggingface/`.

There are two model flows:

- Default inference uses the original HuggingFace `safetensors` checkpoint.
- Persistent quantized inference uses a local `GGUF` directory produced by `qwencandle quantize` and selected with `--model /path/to/dir`.

### Options

```
--model <id>       HuggingFace model ID or local path (default: Qwen/Qwen3-ASR-0.6B)
--device <dev>     Device: cpu, metal, cuda (default: auto-detect)
--language <lang>  Force output language (e.g. English, Chinese, Japanese)
--context <text>   Condition on previous text (system prompt for consistency)
--help             Show help
```

### Quantize

Create a persistent quantized `GGUF` model directory:

```
cargo run --release --features metal -- quantize \
  --src Qwen/Qwen3-ASR-0.6B \
  --dst ./qwen3-asr-q8_0 \
  --dtype q8_0
```

Then run inference against the exported model:

```
ffmpeg -i audio.mp3 -ac 1 -ar 16000 -f wav -acodec pcm_f32le - | \
  ./target/release/qwencandle --model ./qwen3-asr-q8_0 --device cpu
```

The generated directory contains:

- `model.gguf`
- tokenizer files copied from the source model (`tokenizer.json` or `vocab.json` + `merges.txt`)

Supported quantization dtypes:

- Recommended for this model: `q8_0`, `q5_0`, `q4_0`
- Also supported by the CLI: `q4k`, `q5k`, `q6k`, `q8k`

Important: the `K` formats can fail on this model. Some Qwen3-ASR-0.6B linear weights have last dimension `896`, and Candle requires the quantized tensor's last dimension to be divisible by the format block size. The exporter fails explicitly instead of silently falling back. For example:

```
qwencandle quantize --src Qwen/Qwen3-ASR-0.6B --dst ./bad --dtype q6k
# Error: ... last dim 896 is not divisible by block size 256
```

Important: the default CPU command below does not use the quantized model:

```
cat fixtures/jfk.wav | cargo run --release --features metal -- --device cpu
```

That command still loads the default `Qwen/Qwen3-ASR-0.6B` safetensors checkpoint. To run the exported quantized `GGUF`, you must pass `--model` pointing at the quantized directory:

```
cat fixtures/jfk.wav | cargo run --release --features metal -- \
  --model ./qwen3-asr-q8_0 --device cpu
```

### CPU acceleration

On Linux x86_64, build with `--features mkl` to enable Intel MKL for significantly faster CPU inference. MKL is statically linked — no runtime installation required. It works on both Intel and AMD x86_64 CPUs (on AMD, set `MKL_DEBUG_CPU_TYPE=5` to unlock full AVX2/AVX-512 codepaths). The `mkl` feature requires x86_64 and will fail to compile on other architectures.

On macOS, the `accelerate` feature uses Apple's Accelerate framework for the same purpose.

### Thread count

Without MKL, set `RAYON_NUM_THREADS` to control CPU parallelism for Candle CPU kernels and the mel/STFT preprocessing stage:

```
RAYON_NUM_THREADS=4 ffmpeg -i audio.mp3 -ac 1 -ar 16000 -f wav -acodec pcm_f32le - | ./target/release/qwencandle
```

Defaults to all available cores if unset.

For quantized `GGUF` CPU inference, thread count is also controlled by `RAYON_NUM_THREADS`. The quantized CPU kernels use Candle's Rayon-based path rather than MKL.

With MKL, thread count is controlled via `MKL_NUM_THREADS` or `OMP_NUM_THREADS` instead:

```
MKL_NUM_THREADS=4 ffmpeg -i audio.mp3 -ac 1 -ar 16000 -f wav -acodec pcm_f32le - | ./target/release/qwencandle
```

This does not mean every phase of `transcribe()` will keep all threads busy. The decoder in [`src/lib.rs`](src/lib.rs) runs autoregressively one token at a time, and without MKL, Candle's CPU `gemm` backend only fans out once an operation is large enough to cross its internal threading threshold. On CPU, it is normal to see some phases use fewer threads than configured.

### Metal GPU

On macOS, build with `--features metal,accelerate` for GPU acceleration and optimized CPU BLAS. The device is auto-detected, or use `--device metal` explicitly:

```
ffmpeg -i audio.mp3 -ac 1 -ar 16000 -f wav -acodec pcm_f32le - | ./target/release/qwencandle
ffmpeg -i audio.mp3 -ac 1 -ar 16000 -f wav -acodec pcm_f32le - | ./target/release/qwencandle --device metal
```

### Language forcing

By default the model auto-detects the spoken language. Use `--language` to force a specific output language:

```
ffmpeg -i audio.mp3 -ac 1 -ar 16000 -f wav -acodec pcm_f32le - | ./target/release/qwencandle --language Japanese
```

Supported languages: Chinese, English, Cantonese, Arabic, German, French, Spanish, Portuguese, Indonesian, Italian, Korean, Russian, Thai, Vietnamese, Japanese, Turkish, Hindi, Malay, Dutch, Swedish, Danish, Finnish, Polish, Czech, Filipino, Persian, Greek, Romanian, Hungarian, Macedonian.

### Context conditioning

Pass previous transcript text to improve consistency across segments:

```
ffmpeg -i segment2.wav -ac 1 -ar 16000 -f wav -acodec pcm_f32le - | ./target/release/qwencandle --context "Previously the speaker discussed climate change."
```

### Local model

To use a locally downloaded model instead of auto-downloading:

```
huggingface-cli download Qwen/Qwen3-ASR-0.6B --local-dir qwen3-asr-0.6b
ffmpeg -i audio.mp3 -ac 1 -ar 16000 -f wav -acodec pcm_f32le - | ./target/release/qwencandle --model ./qwen3-asr-0.6b
```

To use a local quantized GGUF directory:

```
ffmpeg -i audio.mp3 -ac 1 -ar 16000 -f wav -acodec pcm_f32le - | \
  ./target/release/qwencandle --model ./qwen3-asr-q8_0 --device cpu
```

## Rust library

```rust
use qwencandle::{QwenAsr, Device, best_device, is_cuda_available, is_metal_available};

// Auto-detect best device (CUDA > Metal > CPU)
let device = best_device()?;
let mut model = QwenAsr::load_on("Qwen/Qwen3-ASR-0.6B", &device)?;
let text = model.transcribe(&samples, Some("English"), None)?;

// Check device availability (like torch.cuda.is_available())
if is_cuda_available() { /* ... */ }
if is_metal_available() { /* ... */ }
```

## Python bindings

### Install from GitHub Pages

change `VERSION` to the version you want to install (e.g. `0.0.1a1`):

```
pip install --extra-index-url https://andrewtheguy.github.io/qwencandle/simple/ qwencandle==VERSION
```

Or with uv:

```
uv pip install --extra-index-url https://andrewtheguy.github.io/qwencandle/simple/ qwencandle==VERSION
```

### Install from source

```
uv venv
uv pip install numpy maturin
maturin develop --release
```

With Metal GPU + Accelerate BLAS (macOS):

```
maturin develop --release --features metal,accelerate
```

With Intel MKL BLAS (Linux x86_64):

```
maturin develop --release --features mkl
```

### Usage

```python
import numpy as np
import qwencandle

# Check device availability (like torch.cuda.is_available())
qwencandle.is_cuda_available()   # True if CUDA compiled and available
qwencandle.is_metal_available()  # True if Metal compiled and available

# Device is required
model = qwencandle.QwenAsr("cpu")  # auto-downloads from HuggingFace
text = model.transcribe(samples)   # samples: numpy float32 array, 16kHz mono

# with options
model = qwencandle.QwenAsr("metal", model_id="Qwen/Qwen3-ASR-0.6B")
text = model.transcribe(samples, language="English", context="Previous sentence.")

# local quantized GGUF directory
model = qwencandle.QwenAsr("cpu", model_id="./qwen3-asr-q8_0")

# direct GGUF file path also works if tokenizer files are alongside it
model = qwencandle.QwenAsr("cpu", model_id="./qwen3-asr-q8_0/model.gguf")
```

### API

```python
qwencandle.DEFAULT_MODEL_ID      # "Qwen/Qwen3-ASR-0.6B"
qwencandle.SUPPORTED_LANGUAGES   # list of 30 language names
qwencandle.is_cuda_available()   # bool
qwencandle.is_metal_available()  # bool

class QwenAsr:
    # model_id may be a HuggingFace model ID, local safetensors directory,
    # local GGUF directory, or direct .gguf file path
    def __init__(self, device: str, model_id: str | None = None): ...
    def transcribe(self, samples, *, language=None, context=None) -> str: ...
```

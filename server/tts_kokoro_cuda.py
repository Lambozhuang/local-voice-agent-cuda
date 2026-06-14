#
# In-process Kokoro TTS service for NVIDIA CUDA.
#
# On macOS, Kokoro was run through a subprocess worker (tts_mlx_isolated.py +
# kokoro_worker.py) purely to dodge Apple Metal threading conflicts. CUDA has no
# such constraint, so here we load the PyTorch `kokoro` KPipeline directly in the
# bot process and run it on the GPU.
#

import asyncio
from typing import AsyncGenerator, Optional

import numpy as np
from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts


class KokoroCUDATTSService(TTSService):
    """Local Kokoro TTS running in-process on an NVIDIA GPU via the PyTorch `kokoro` package."""

    def __init__(
        self,
        *,
        voice: str = "af_heart",
        lang_code: str = "a",
        device: str = "cuda",
        sample_rate: int = 24000,
        **kwargs,
    ):
        """Initialize the Kokoro CUDA TTS service.

        Args:
            voice: Kokoro voice name (e.g. "af_heart").
            lang_code: Kokoro language code ("a" = American English).
            device: Torch device. Defaults to "cuda" to keep inference on the GPU,
                matching the macOS Metal behavior. Pass "cpu" only as a fallback.
            sample_rate: Output sample rate. Kokoro generates at 24 kHz.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._voice = voice
        self._lang_code = lang_code
        self._device = device

        self._pipeline = None
        self._initialized = False

        self._settings = {
            "voice": voice,
            "lang_code": lang_code,
            "device": device,
            "sample_rate": sample_rate,
        }

    def _load_pipeline(self):
        """Load the Kokoro KPipeline onto the GPU. Runs in an executor thread."""
        import torch
        from kokoro import KPipeline

        if self._device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested for Kokoro TTS but torch.cuda.is_available() is False. "
                "Install a CUDA-enabled PyTorch build and ensure an NVIDIA GPU is visible."
            )

        pipeline = KPipeline(lang_code=self._lang_code, device=self._device)
        logger.info(f"Loaded Kokoro pipeline on device: {self._device}")
        return pipeline

    async def _initialize_if_needed(self) -> bool:
        if self._initialized:
            return True

        loop = asyncio.get_event_loop()
        try:
            self._pipeline = await loop.run_in_executor(None, self._load_pipeline)
            self._initialized = True
            logger.info("Kokoro CUDA TTS initialized")
            return True
        except Exception as e:
            logger.error(f"Kokoro initialization failed: {e}")
            return False

    def _generate_audio(self, text: str) -> Optional[np.ndarray]:
        """Synchronously run Kokoro and return float32 audio in [-1, 1]. Executor thread."""
        segments = []
        for result in self._pipeline(text, voice=self._voice, speed=1.0):
            # KPipeline yields results whose `.audio` is a torch tensor.
            audio = result.audio
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().numpy()
            segments.append(np.asarray(audio, dtype=np.float32))

        if not segments:
            return None
        return segments[0] if len(segments) == 1 else np.concatenate(segments, axis=0)

    def can_generate_metrics(self) -> bool:
        return True

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech in-process on the GPU."""
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            await self.start_ttfb_metrics()
            await self.start_tts_usage_metrics(text)

            yield TTSStartedFrame()

            if not await self._initialize_if_needed():
                raise RuntimeError("Failed to initialize Kokoro CUDA TTS")

            loop = asyncio.get_event_loop()
            audio = await loop.run_in_executor(None, self._generate_audio, text)

            if audio is None:
                raise RuntimeError("No audio generated")

            if np.max(np.abs(audio)) < 1e-6:
                raise RuntimeError("Generated audio is silent")

            # Convert to 16-bit PCM.
            audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()

            await self.stop_ttfb_metrics()

            CHUNK_SIZE = self.chunk_size
            for i in range(0, len(audio_bytes), CHUNK_SIZE):
                chunk = audio_bytes[i : i + CHUNK_SIZE]
                if len(chunk) > 0:
                    yield TTSAudioRawFrame(chunk, self.sample_rate, 1)
                    await asyncio.sleep(0.001)

        except Exception as e:
            logger.error(f"Error in run_tts: {e}")
            yield ErrorFrame(error=str(e))
        finally:
            logger.debug(f"{self}: Finished TTS [{text}]")
            await self.stop_ttfb_metrics()
            yield TTSStoppedFrame()

    async def __aenter__(self):
        await super().__aenter__()
        await self._initialize_if_needed()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await super().__aexit__(exc_type, exc_val, exc_tb)

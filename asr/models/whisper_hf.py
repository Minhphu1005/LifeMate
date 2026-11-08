"""Adapter Whisper / PhoWhisper (HuggingFace transformers). Greedy, offline, tất định."""
from __future__ import annotations

from typing import Iterable, List, Optional

from ..base import BaseASR
from ..normalize import postprocess


def _load_audio(path: str, sr: int = 16000):
    import soundfile as sf
    audio, rate = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=rate, target_sr=sr)
    return audio


class WhisperASR(BaseASR):
    def __init__(self, model_path: str, language: str = "vi", device: Optional[str] = None,
                 fp16: bool = True, batch_size: int = 8, num_beams: int = 1,
                 prompt: str = "", digit_sequences: bool = True, seed: int = 42,
                 max_new_tokens: int = 128, no_repeat_ngram_size: int = 4, **_):
        import random
        import numpy as np
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if (fp16 and self.device == "cuda") else torch.float32
        self.processor = WhisperProcessor.from_pretrained(model_path)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=dtype).to(self.device).eval()
        self.dtype = dtype
        self.language = language
        self.batch_size = batch_size
        self.num_beams = num_beams  # 1 = greedy
        self.max_new_tokens = max_new_tokens
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.digit_sequences = digit_sequences
        self.prompt_ids = (self.processor.get_prompt_ids(prompt, return_tensors="pt").to(self.device)
                           if prompt else None)

    def _generate(self, arrays: list) -> List[str]:
        feats = self.processor.feature_extractor(
            arrays, sampling_rate=16000, return_tensors="pt").input_features
        feats = feats.to(self.device, dtype=self.dtype)
        kwargs = dict(language=self.language, task="transcribe", do_sample=False,
                      num_beams=self.num_beams, max_new_tokens=self.max_new_tokens)
        if self.no_repeat_ngram_size:  # chống lặp từ vô tận ("bài bài bài ...")
            kwargs["no_repeat_ngram_size"] = self.no_repeat_ngram_size
        if self.prompt_ids is not None:
            kwargs["prompt_ids"] = self.prompt_ids
        with self.torch.inference_mode():
            ids = self.model.generate(feats, **kwargs)
        texts = self.processor.batch_decode(ids, skip_special_tokens=True)
        # Whisper có thể lặp lại prompt ở đầu output -> bỏ nếu có
        return [postprocess(self._strip_prompt(t), self.digit_sequences) for t in texts]

    def _strip_prompt(self, text: str) -> str:
        return text

    def transcribe(self, audio_path: str, metadata: Optional[dict] = None) -> str:
        return self._generate([_load_audio(audio_path)])[0]

    def transcribe_batch(self, audio_paths: Iterable[str],
                         metadata: Optional[dict] = None) -> List[str]:
        paths = list(audio_paths)
        out: List[str] = []
        for i in range(0, len(paths), self.batch_size):
            out.extend(self._generate([_load_audio(p) for p in paths[i:i + self.batch_size]]))
        return out


"""Adapter Whisper / PhoWhisper (HuggingFace transformers).
- Greedy decoding, offline, tất định (seed cố định).
- Tự động cắt khoảng lặng đầu/cuối (Silence Trimming) giúp tăng tốc và chống ảo giác cuối băng.
- Hỗ trợ bơm từ khóa ngữ cảnh động (Dynamic Context Prompting) qua metadata["prompt"].
"""
from __future__ import annotations

from typing import Iterable, List, Optional

from ..base import BaseASR
from ..normalize import postprocess


def _trim_silence(audio, sr: int = 16000, threshold: float = 0.008, pad_s: float = 0.15):
    """Cắt khoảng lặng ở đầu và cuối file âm thanh, giữ lại đệm an toàn pad_s giây."""
    import numpy as np
    if audio.size < int(sr * 0.6):
        return audio
    mask = np.abs(audio) > threshold
    if not mask.any():
        return audio
    idx = np.flatnonzero(mask)
    pad = int(sr * pad_s)
    start = max(0, int(idx[0]) - pad)
    end = min(int(audio.size), int(idx[-1]) + 1 + pad)
    if end - start < int(sr * 0.5):
        return audio
    return audio[start:end]


def _load_audio(path: str, sr: int = 16000, trim_silence: bool = True):
    import soundfile as sf
    audio, rate = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=rate, target_sr=sr)
    if trim_silence:
        audio = _trim_silence(audio, sr=sr)
    return audio


class WhisperASR(BaseASR):
    def __init__(self, model_path: str, language: str = "vi", device: Optional[str] = None,
                 fp16: bool = True, batch_size: int = 8, num_beams: int = 1,
                 prompt: str = "", digit_sequences: bool = True, trim_silence: bool = True,
                 seed: int = 42, max_new_tokens: int = 128, no_repeat_ngram_size: int = 4, **_):
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
        if hasattr(self.model, "generation_config") and self.model.generation_config is not None:
            self.model.generation_config.max_length = None
        self.dtype = dtype
        self.language = language
        self.batch_size = batch_size
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.digit_sequences = digit_sequences
        self.trim_silence = trim_silence
        self.default_prompt = prompt

    def _encode_prompt(self, prompt_text: str):
        if not prompt_text:
            return None
        try:
            return self.processor.get_prompt_ids(prompt_text, return_tensors="pt").to(self.device)
        except Exception:
            return None

    def _generate(self, arrays: list, prompt_text: str = "") -> List[str]:
        inputs = self.processor.feature_extractor(
            arrays, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
        feats = inputs.input_features.to(self.device, dtype=self.dtype)
        attn_mask = inputs.attention_mask.to(self.device) if hasattr(inputs, "attention_mask") and inputs.attention_mask is not None else None
        kwargs = dict(language=self.language, task="transcribe", do_sample=False,
                      num_beams=self.num_beams, max_new_tokens=self.max_new_tokens)
        if attn_mask is not None:
            kwargs["attention_mask"] = attn_mask
        if self.no_repeat_ngram_size:
            kwargs["no_repeat_ngram_size"] = self.no_repeat_ngram_size
        pids = self._encode_prompt(prompt_text or self.default_prompt)
        if pids is not None:
            kwargs["prompt_ids"] = pids
        with self.torch.inference_mode():
            ids = self.model.generate(feats, **kwargs)
        texts = self.processor.batch_decode(ids, skip_special_tokens=True)
        return [postprocess(self._strip_prompt(t, prompt_text or self.default_prompt), self.digit_sequences)
                for t in texts]

    def _strip_prompt(self, text: str, prompt_text: str) -> str:
        if prompt_text and text.lower().startswith(prompt_text.lower()):
            return text[len(prompt_text):].lstrip(" ,.:;-")
        return text

    def transcribe(self, audio_path: str, metadata: Optional[dict] = None) -> str:
        p = (metadata or {}).get("prompt", "")
        return self._generate([_load_audio(audio_path, trim_silence=self.trim_silence)], prompt_text=p)[0]

    def transcribe_batch(self, audio_paths: Iterable[str],
                         metadata: Optional[dict] = None) -> List[str]:
        paths = list(audio_paths)
        if not paths:
            return []
        p = (metadata or {}).get("prompt", "")
        out: List[str] = []
        for i in range(0, len(paths), self.batch_size):
            chunk = [_load_audio(x, trim_silence=self.trim_silence) for x in paths[i:i + self.batch_size]]
            out.extend(self._generate(chunk, prompt_text=p))
        return out

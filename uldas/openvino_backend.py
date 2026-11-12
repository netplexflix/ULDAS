"""Whisper on Intel GPUs (Arc, Iris Xe, UHD) through OpenVINO.

CTranslate2, the engine behind faster-whisper, runs on CPU and NVIDIA CUDA
only.  This module runs the OpenVINO exports of the same Whisper checkpoints
(https://huggingface.co/OpenVINO) and exposes the two methods ULDAS calls on
a faster-whisper ``WhisperModel``:

* ``detect_language(audio=...)`` -> ``(language, probability, all_probs)``
* ``transcribe(audio, language=..., ...)`` -> ``(segments, info)``

It needs only the ``openvino`` runtime package: audio features and tokens
come from faster-whisper's own (NumPy) feature extractor and tokenizer, so no
PyTorch or transformers install is required.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 30 * SAMPLE_RATE
N_FRAMES = 3000
MAX_NEW_TOKENS = 224
TIME_PRECISION = 0.02   # seconds per timestamp token

# ULDAS/faster-whisper model names -> OpenVINO Hugging Face exports
_OV_MODELS = {
    "tiny": "whisper-tiny", "tiny.en": "whisper-tiny.en",
    "base": "whisper-base", "base.en": "whisper-base.en",
    "small": "whisper-small", "small.en": "whisper-small.en",
    "medium": "whisper-medium", "medium.en": "whisper-medium.en",
    "large-v2": "whisper-large-v2",
    "large": "whisper-large-v3", "large-v3": "whisper-large-v3",
    "turbo": "whisper-large-v3-turbo", "large-v3-turbo": "whisper-large-v3-turbo",
    "distil-large-v2": "distil-whisper-large-v2",
    "distil-large-v3": "distil-whisper-large-v3",
}
_PRECISIONS = {"float16": "fp16", "fp16": "fp16", "float32": "fp16",
               "int8": "int8", "int8_float16": "int8", "int8_float32": "int8",
               "int4": "int4"}


def is_openvino_device(device: str) -> bool:
    return (device or "").lower().split(":")[0] in ("openvino", "intel")


def openvino_target(device: str) -> str:
    """``openvino`` / ``intel`` -> ``GPU``; ``openvino:GPU.1`` -> ``GPU.1``."""
    _, _, target = (device or "").partition(":")
    return target.upper() if target else "GPU"


def openvino_gpu_available() -> Optional[str]:
    """Name of the first Intel GPU OpenVINO can use, or None."""
    try:
        import openvino as ov
        core = ov.Core()
        for dev in core.available_devices:
            if dev.startswith("GPU"):
                return core.get_property(dev, "FULL_DEVICE_NAME")
    except Exception:
        pass
    return None


def resolve_model(name: str, compute_type: str) -> str:
    """Map a model setting to a local directory or an OpenVINO HF repo id."""
    if os.path.isdir(name) or "/" in name:
        return name
    precision = _PRECISIONS.get((compute_type or "auto").lower(), "fp16")
    base = _OV_MODELS.get(name)
    if base is None:
        raise ValueError(f"No OpenVINO export known for Whisper model '{name}'. "
                         f"Use one of {', '.join(sorted(_OV_MODELS))}, a Hugging "
                         f"Face repo id or a local OpenVINO model directory.")
    return f"OpenVINO/{base}-{precision}-ov"


class OpenVINOWhisper:
    """Greedy Whisper decoding on an OpenVINO device."""

    def __init__(self, model: str, device: str = "GPU", compute_type: str = "auto",
                 cache_dir: Optional[str] = None, local_files_only: bool = False):
        import openvino as ov
        from faster_whisper.feature_extractor import FeatureExtractor
        from faster_whisper.tokenizer import _LANGUAGE_CODES
        from tokenizers import Tokenizer

        repo = resolve_model(model, compute_type)
        if os.path.isdir(repo):
            path = Path(repo)
        else:
            from huggingface_hub import snapshot_download
            path = Path(snapshot_download(repo, local_files_only=local_files_only))
        self.model_path = path
        self.repo = repo

        cfg = json.loads((path / "config.json").read_text())
        gen = json.loads((path / "generation_config.json").read_text()) \
            if (path / "generation_config.json").exists() else {}
        self.feature_extractor = FeatureExtractor(feature_size=cfg.get("num_mel_bins", 80))
        self.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))

        tid = self.tokenizer.token_to_id
        self.sot = tid("<|startoftranscript|>")
        self.eot = tid("<|endoftext|>")
        self.transcribe_id = tid("<|transcribe|>")
        self.translate_id = tid("<|translate|>")
        self.no_timestamps = tid("<|notimestamps|>")
        self.ts_begin = self.no_timestamps + 1
        self.no_speech = tid("<|nospeech|>") or tid("<|nocaptions|>")
        self.lang_ids = {code: tid(f"<|{code}|>") for code in _LANGUAGE_CODES
                         if tid(f"<|{code}|>") is not None}
        self.is_multilingual = bool(gen.get("is_multilingual", len(self.lang_ids) > 1))
        constant_suppress = set(gen.get("suppress_tokens", []))
        constant_suppress.update(range(self.eot + 1, self.ts_begin))
        self.constant_suppress = np.array(sorted(constant_suppress), dtype=np.int64)
        self.begin_suppress = np.array(gen.get("begin_suppress_tokens", [220, self.eot]),
                                       dtype=np.int64)

        core = ov.Core()
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            core.set_property({"CACHE_DIR": cache_dir})
        self.device = device
        self.device_name = core.get_property(device, "FULL_DEVICE_NAME") \
            if device != "AUTO" else device
        self._encoder = core.compile_model(path / "openvino_encoder_model.xml", device)
        self._decoder = core.compile_model(path / "openvino_decoder_model.xml", device)
        self._enc_req = self._encoder.create_infer_request()
        self._dec_req = self._decoder.create_infer_request()
        self._dec_inputs = {i.get_any_name() for i in self._decoder.inputs}

    # ── internals ────────────────────────────────────────────────────────
    def _features(self, audio: np.ndarray) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32)
        if len(audio) < CHUNK_SAMPLES:
            chunk = np.pad(audio, (0, CHUNK_SAMPLES - len(audio)))
        else:
            chunk = audio[:CHUNK_SAMPLES]
        mel = self.feature_extractor(chunk)[:, :N_FRAMES]
        if mel.shape[1] < N_FRAMES:
            mel = np.pad(mel, ((0, 0), (0, N_FRAMES - mel.shape[1])))
        return mel[None].astype(np.float32)

    def _encode(self, audio: np.ndarray) -> np.ndarray:
        self._enc_req.infer({"input_features": self._features(audio)})
        return self._enc_req.get_output_tensor(0).data.copy()

    def _decode_step(self, ids: List[int], enc: np.ndarray) -> np.ndarray:
        """Feed *ids* (appended to the cached sequence); returns logits [len(ids), V]."""
        inputs = {"input_ids": np.array([ids], dtype=np.int64),
                  "encoder_hidden_states": enc}
        if "beam_idx" in self._dec_inputs:
            inputs["beam_idx"] = np.zeros(1, dtype=np.int32)
        self._dec_req.infer(inputs)
        return self._dec_req.get_output_tensor(0).data[0].astype(np.float32)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()

    def _language_probs(self, enc: np.ndarray) -> List[Tuple[str, float]]:
        self._dec_req.reset_state()
        logits = self._decode_step([self.sot], enc)[-1]
        codes = list(self.lang_ids)
        probs = self._softmax(logits[[self.lang_ids[c] for c in codes]])
        return sorted(zip(codes, probs.tolist()), key=lambda kv: kv[1], reverse=True)

    # ── faster-whisper compatible API ────────────────────────────────────
    def detect_language(self, audio: Optional[np.ndarray] = None, features=None, **_):
        if audio is None:
            raise ValueError("OpenVINO backend needs raw audio for detect_language")
        if not self.is_multilingual:
            return "en", 1.0, [("en", 1.0)]
        all_probs = self._language_probs(self._encode(audio))
        return all_probs[0][0], all_probs[0][1], all_probs

    def transcribe(self, audio: np.ndarray, language: Optional[str] = None,
                   task: str = "transcribe", repetition_penalty: float = 1.0,
                   no_repeat_ngram_size: int = 0, log_prob_threshold: Optional[float] = -1.0,
                   no_speech_threshold: Optional[float] = 0.6,
                   without_timestamps: bool = False, max_initial_timestamp: float = 1.0,
                   **_):
        """Greedy decoding of *audio*, seeking through it like faster-whisper.

        Honours the options that shape a greedy transcript (language, task,
        repetition_penalty, no_repeat_ngram_size, the no-speech / log-prob
        skip, timestamps on or off).  Beam search, temperature fallback,
        prompts and word timestamps are not implemented: ULDAS decodes with
        beam_size=1, temperature=0 and no prompt.
        """
        audio = np.asarray(audio, dtype=np.float32)
        total = len(audio)
        duration = total / SAMPLE_RATE
        segments, lang_prob, seek = [], 1.0, 0
        while seek < total - SAMPLE_RATE // 10:
            chunk = audio[seek:seek + CHUNK_SAMPLES]
            chunk_len = len(chunk)
            enc = self._encode(chunk)
            if language is None and self.is_multilingual:
                language, lang_prob = self._language_probs(enc)[0]
            tokens, avg_logprob, no_speech_prob = self._greedy(
                enc, language or "en", task, repetition_penalty, no_repeat_ngram_size,
                not without_timestamps, max_initial_timestamp)
            if (no_speech_threshold is not None and no_speech_prob > no_speech_threshold
                    and (log_prob_threshold is None or avg_logprob < log_prob_threshold)):
                seek += chunk_len
                continue
            offset = seek / SAMPLE_RATE
            pieces, advance = self._split_segments(tokens, chunk_len)
            for start, end, toks in pieces:
                text = re.sub(r"\s+", " ", self.tokenizer.decode(toks, skip_special_tokens=True))
                if text.strip():
                    segments.append(SimpleNamespace(
                        start=round(offset + start, 2),
                        end=round(min(duration, offset + end), 2),
                        text=text if text.startswith(" ") else " " + text,
                        avg_logprob=avg_logprob, no_speech_prob=no_speech_prob))
            seek += max(advance, SAMPLE_RATE // 50)
        info = SimpleNamespace(language=language, language_probability=lang_prob,
                               duration=duration, all_language_probs=None)
        return iter(segments), info

    def _split_segments(self, tokens: List[int], chunk_len: int):
        """Split decoded tokens at timestamp pairs (Whisper's own rules).

        Returns ``([(start_s, end_s, text_tokens)], samples_to_advance)``.
        """
        tb = self.ts_begin

        def to_s(t):
            return (t - tb) * TIME_PRECISION

        is_ts = [t >= tb for t in tokens]
        single_ending = len(tokens) >= 2 and is_ts[-1] and not is_ts[-2]
        cuts = [i for i in range(1, len(tokens)) if is_ts[i] and is_ts[i - 1]]
        if not cuts:
            # One segment (or no timestamps at all): take the whole window.
            stamps = [to_s(t) for t in tokens if t >= tb]
            text = [t for t in tokens if t < tb]
            end = stamps[-1] if len(stamps) > 1 else chunk_len / SAMPLE_RATE
            return [(stamps[0] if stamps else 0.0, end, text)], chunk_len
        if single_ending:
            cuts.append(len(tokens))
        pieces, last = [], 0
        for cut in cuts:
            part = tokens[last:cut]
            pieces.append((to_s(part[0]), to_s(part[-1]), [t for t in part if t < tb]))
            last = cut
        if single_ending:
            return pieces, chunk_len
        # Ended mid-segment: resume decoding after the last complete segment.
        return pieces, int(to_s(tokens[last - 1]) * SAMPLE_RATE)

    def _greedy(self, enc, language, task, repetition_penalty, no_repeat_ngram_size,
                timestamps=True, max_initial_timestamp=1.0):
        prompt = [self.sot]
        if self.is_multilingual:
            prompt += [self.lang_ids.get(language, self.lang_ids.get("en")),
                       self.translate_id if task == "translate" else self.transcribe_id]
        if not timestamps:
            prompt.append(self.no_timestamps)

        tb, eot = self.ts_begin, self.eot
        self._dec_req.reset_state()
        logits = self._decode_step(prompt, enc)
        no_speech_prob = float(self._softmax(logits[0])[self.no_speech]) \
            if self.no_speech is not None else 0.0
        step_logits = logits[-1]
        tokens: List[int] = []
        seen_text_tokens = set()
        seen_text_list: List[int] = []
        last_stamp: Optional[int] = None
        logprob_sum = 0.0
        n_logprobs = 0
        for n in range(MAX_NEW_TOKENS):
            x = step_logits.copy()
            x[self.constant_suppress] = -np.inf      # language/task/control & suppress tokens
            if not timestamps:
                x[tb:] = -np.inf
            if n == 0:
                x[self.begin_suppress] = -np.inf
            if repetition_penalty and repetition_penalty != 1.0 and seen_text_list:
                v = x[seen_text_list]
                x[seen_text_list] = np.where(v > 0, v / repetition_penalty, v * repetition_penalty)
            if no_repeat_ngram_size and len(tokens) >= no_repeat_ngram_size:
                k = no_repeat_ngram_size
                tail = tuple(tokens[-(k - 1):]) if k > 1 else ()
                for i in range(len(tokens) - k + 1):
                    if tuple(tokens[i:i + k - 1]) == tail:
                        x[tokens[i + k - 1]] = -np.inf
            if timestamps:
                self._timestamp_rules(x, tokens, n, max_initial_timestamp, last_stamp)
            tok = int(np.argmax(x))
            m = x[tok]
            logprob_sum += float(-np.log(np.exp(x - m).sum()))
            n_logprobs += 1
            if tok == eot:
                break
            tokens.append(tok)
            if tok >= tb:
                last_stamp = tok
            elif tok not in seen_text_tokens:
                seen_text_tokens.add(tok)
                seen_text_list.append(tok)
            step_logits = self._decode_step([tok], enc)[-1]
        return tokens, logprob_sum / max(1, n_logprobs), no_speech_prob

    def _timestamp_rules(self, x: np.ndarray, tokens: List[int], n: int,
                         max_initial_timestamp: float, last_stamp: Optional[int]) -> None:
        """Whisper's ApplyTimestampRules, in place on the logits *x*."""
        tb, eot = self.ts_begin, self.eot
        x[self.no_timestamps] = -np.inf
        last_ts = len(tokens) >= 1 and tokens[-1] >= tb
        penult_ts = len(tokens) < 2 or tokens[-2] >= tb
        if last_ts:
            if penult_ts:
                x[tb:] = -np.inf                     # pair closed: text or EOT next
            else:
                x[:eot] = -np.inf                    # open pair: timestamp or EOT next
        if last_stamp is not None:
            floor = last_stamp if (last_ts and not penult_ts) else last_stamp + 1
            x[tb:floor] = -np.inf                    # timestamps never go backwards
        if n == 0:
            x[:tb] = -np.inf                         # first token is a timestamp
            x[tb + int(round(max_initial_timestamp / TIME_PRECISION)) + 1:] = -np.inf
        # If timestamps are collectively likelier than any text token, emit one.
        ts_logits = x[tb:]
        text_logits = x[:tb]
        ts_finite = np.isfinite(ts_logits)
        text_finite = np.isfinite(text_logits)
        if ts_finite.any() and text_finite.any():
            text_max = float(text_logits[text_finite].max())
            if np.exp(ts_logits[ts_finite] - text_max).sum() > 1.0:
                x[:tb] = -np.inf

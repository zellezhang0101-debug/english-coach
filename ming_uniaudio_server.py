import os
import tempfile
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response, JSONResponse


"""
Ming-UniAudio lightweight HTTP wrapper.

This server is intentionally decoupled from app.py so you can:
- run it locally now (typically on a GPU machine)
- later move it to a remote host without changing the main Flask app

Main Flask app expects:
  POST {MING_UNIAUDIO_URL}/tts  -> returns audio/wav bytes
  POST {MING_UNIAUDIO_URL}/asr  -> returns JSON {"text": "..."} (or {"transcript": "..."})
"""


APP = FastAPI()

# Where you cloned Ming-UniAudio source code (so we can import their Python modules).
MING_UNIAUDIO_REPO_PATH = (os.getenv("MING_UNIAUDIO_REPO_PATH") or "").strip()
# Where the model weights live (downloaded from HuggingFace/ModelScope).
MING_UNIAUDIO_MODEL_PATH = (os.getenv("MING_UNIAUDIO_MODEL_PATH") or "").strip()
MING_UNIAUDIO_DEVICE = (os.getenv("MING_UNIAUDIO_DEVICE") or "cuda:0").strip()

# Prompt voice for TTS (Ming-UniAudio TTS needs a prompt wav + prompt text).
MING_UNIAUDIO_PROMPT_WAV = (os.getenv("MING_UNIAUDIO_PROMPT_WAV") or "").strip()
MING_UNIAUDIO_PROMPT_TEXT = (os.getenv("MING_UNIAUDIO_PROMPT_TEXT") or "").strip()

_ming = None
_ming_load_error: Optional[str] = None


def _lazy_load():
    global _ming, _ming_load_error
    if _ming is not None or _ming_load_error is not None:
        return

    try:
        if not MING_UNIAUDIO_REPO_PATH:
            raise RuntimeError("MING_UNIAUDIO_REPO_PATH is not set")
        if not os.path.isdir(MING_UNIAUDIO_REPO_PATH):
            raise RuntimeError("MING_UNIAUDIO_REPO_PATH does not exist: " + MING_UNIAUDIO_REPO_PATH)
        if not MING_UNIAUDIO_MODEL_PATH:
            raise RuntimeError("MING_UNIAUDIO_MODEL_PATH is not set")

        import sys

        if MING_UNIAUDIO_REPO_PATH not in sys.path:
            sys.path.insert(0, MING_UNIAUDIO_REPO_PATH)

        # Ming-UniAudio code uses these imports as shown in their demo.
        import warnings

        warnings.filterwarnings("ignore")

        import torch  # noqa: F401
        from transformers import AutoProcessor  # noqa: F401

        from modeling_bailingmm import BailingMMNativeForConditionalGeneration

        class MingAudioWrapper:
            def __init__(self, model_path, device="cuda:0"):
                import torch
                from transformers import AutoProcessor

                self.device = device
                self.model = (
                    BailingMMNativeForConditionalGeneration.from_pretrained(
                        model_path,
                        torch_dtype=torch.bfloat16,
                        low_cpu_mem_usage=True,
                    )
                    .eval()
                    .to(torch.bfloat16)
                    .to(self.device)
                )
                # Processor expects to be loaded from Ming-UniAudio repo root.
                self.processor = AutoProcessor.from_pretrained(MING_UNIAUDIO_REPO_PATH, trust_remote_code=True)
                self.tokenizer = self.processor.tokenizer
                self.sample_rate = self.processor.audio_processor.sample_rate
                self.patch_size = self.processor.audio_processor.patch_size

            def asr(self, audio_path: str) -> str:
                messages = [
                    {
                        "role": "HUMAN",
                        "content": [
                            {
                                "type": "text",
                                "text": "Please recognize the language of this speech and transcribe it. Format: oral.",
                            },
                            {"type": "audio", "audio": audio_path},
                        ],
                    },
                ]
                text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
                image_inputs, video_inputs, audio_inputs = self.processor.process_vision_info(messages)
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    audios=audio_inputs,
                    return_tensors="pt",
                ).to(self.device)

                import torch

                for k in list(inputs.keys()):
                    if k in ("pixel_values", "pixel_values_videos", "audio_feats"):
                        inputs[k] = inputs[k].to(dtype=torch.bfloat16)

                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    eos_token_id=self.processor.gen_terminator,
                )
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                return (output_text or "").strip()

            def tts(self, text: str, *, prompt_wav_path: str, prompt_text: str, lang: str = "en") -> bytes:
                if not prompt_wav_path or not prompt_text:
                    raise RuntimeError("TTS requires prompt_wav_path + prompt_text (set env vars or pass in request).")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tf:
                    out_path = tf.name
                try:
                    # Ming-UniAudio provides generate_tts on the model.
                    self.model.generate_tts(
                        text=text,
                        prompt_wav_path=prompt_wav_path,
                        prompt_text=prompt_text,
                        patch_size=self.patch_size,
                        tokenizer=self.tokenizer,
                        lang=lang,
                        output_wav_path=out_path,
                        sample_rate=self.sample_rate,
                        device=self.device,
                    )
                    with open(out_path, "rb") as f:
                        return f.read()
                finally:
                    try:
                        os.unlink(out_path)
                    except Exception:
                        pass

        _ming = MingAudioWrapper(MING_UNIAUDIO_MODEL_PATH, device=MING_UNIAUDIO_DEVICE)
    except Exception as e:
        _ming_load_error = str(e)


@APP.get("/health")
def health():
    _lazy_load()
    if _ming is None:
        return JSONResponse({"ok": False, "error": _ming_load_error or "not loaded"}, status_code=503)
    return {"ok": True}


@APP.post("/asr")
async def asr(audio: UploadFile = File(...)):
    _lazy_load()
    if _ming is None:
        raise HTTPException(status_code=503, detail=_ming_load_error or "Ming-UniAudio not available")

    suffix = os.path.splitext(audio.filename or "")[-1].lower() or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
        path = tf.name
        tf.write(await audio.read())
    try:
        text = _ming.asr(path)
        if not text:
            raise HTTPException(status_code=500, detail="Empty ASR transcript")
        return {"text": text}
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass


@APP.post("/tts")
async def tts(payload: dict):
    _lazy_load()
    if _ming is None:
        raise HTTPException(status_code=503, detail=_ming_load_error or "Ming-UniAudio not available")

    text = (payload.get("text") or "").strip()
    lang = (payload.get("lang") or "en").strip()
    prompt_wav_path = (payload.get("prompt_wav_path") or MING_UNIAUDIO_PROMPT_WAV).strip()
    prompt_text = (payload.get("prompt_text") or MING_UNIAUDIO_PROMPT_TEXT).strip()

    if not text:
        raise HTTPException(status_code=400, detail="Missing text")

    try:
        wav = _ming.tts(text, prompt_wav_path=prompt_wav_path, prompt_text=prompt_text, lang=lang)
        return Response(wav, media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


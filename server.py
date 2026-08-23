import asyncio
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from transformers import pipeline

SAMPLE_RATE = 16000
WINDOW_SECONDS = 1.0
STEP_SECONDS = 0.8
WINDOW_BYTES = int(SAMPLE_RATE * WINDOW_SECONDS) * 2
STEP_BYTES = int(SAMPLE_RATE * STEP_SECONDS) * 2

speech_pipe = None
inference_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global speech_pipe
    speech_pipe = pipeline(
        "automatic-speech-recognition",
        model="nvidia/parakeet-tdt-0.6b-v3",
        device="cuda",
        dtype=torch.float16,
    )
    yield


app = FastAPI(title="Meeting AI Low-Latency STT", lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "meeting-ai-stt-low-latency",
        "model": "nvidia/parakeet-tdt-0.6b-v3",
        "gpu": "A10G",
        "window_seconds": WINDOW_SECONDS,
        "step_seconds": STEP_SECONDS,
    }


def decode_pcm16(audio_bytes: bytes):
    pcm = np.frombuffer(audio_bytes, dtype=np.int16)
    return np.nan_to_num(pcm.astype(np.float32) / 32768.0)


def transcribe(audio):
    result = speech_pipe({"raw": audio, "sampling_rate": SAMPLE_RATE})
    return result.get("text", "").strip()


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    await ws.send_json({
        "type": "connected",
        "message": "Low-latency STT ready",
        "model": "nvidia/parakeet-tdt-0.6b-v3",
    })
    pcm_buffer = bytearray()
    try:
        while True:
            pcm_buffer.extend(await ws.receive_bytes())
            while len(pcm_buffer) >= WINDOW_BYTES:
                audio = decode_pcm16(bytes(pcm_buffer[:WINDOW_BYTES]))
                del pcm_buffer[:STEP_BYTES]
                try:
                    async with inference_lock:
                        text = await asyncio.to_thread(transcribe, audio)
                    await ws.send_json({
                        "type": "transcript",
                        "text": text,
                        "window_seconds": WINDOW_SECONDS,
                        "step_seconds": STEP_SECONDS,
                    })
                except Exception as exc:
                    await ws.send_json({"type": "error", "message": str(exc)})
    except WebSocketDisconnect:
        pass


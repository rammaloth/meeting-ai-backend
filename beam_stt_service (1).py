import io

from beam import Image, asgi

image = Image(
    python_version="python3.11",
    python_packages=[
        "torch==2.7.1",
        "transformers",
        "accelerate",
        "av",
        "numpy",
        "librosa",
        "fastapi",
        "uvicorn",
    ],
)


def load_model():
    import torch
    from transformers import pipeline

    print("=" * 70)
    print("Loading NVIDIA Parakeet TDT 0.6B v3...")
    print("=" * 70)

    pipe = pipeline(
        "automatic-speech-recognition",
        model="nvidia/parakeet-tdt-0.6b-v3",
        device="cuda",
        dtype=torch.float16,
    )

    print("✅ Parakeet loaded on A10G")
    return pipe


def decode_pcm16_to_float32(audio_bytes):
    """
    Decode raw mono PCM16 audio at 16 kHz into
    float32 samples for Parakeet.
    """
    import numpy as np

    if not audio_bytes:
        return np.zeros(1600, dtype=np.float32), 16000

    pcm = np.frombuffer(
        audio_bytes,
        dtype=np.int16
    )

    if pcm.size == 0:
        return np.zeros(1600, dtype=np.float32), 16000

    audio = (
        pcm.astype(np.float32) / 32768.0
    )

    audio = np.nan_to_num(
        audio,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return audio, 16000


@asgi(
    name="meeting-ai-stt",
    gpu="A10G",
    cpu=2,
    memory="8Gi",
    image=image,
    on_start=load_model,
    concurrent_requests=10,
    keep_warm_seconds=300,
    timeout=3600,
    authorized=False,
)
def web_server(context):
    from fastapi import (
        FastAPI,
        WebSocket,
        WebSocketDisconnect,
    )

    app = FastAPI(
        title="Meeting AI Live STT"
    )

    speech_pipe = context.on_start_value

    # Low-latency rolling transcription.
    # Infer every 0.8 seconds over a 1.0-second window. The 0.2-second
    # overlap protects words that cross chunk boundaries; the client
    # removes repeated boundary words before rendering.
    SAMPLE_RATE = 16000
    WINDOW_SECONDS = 1.0
    STEP_SECONDS = 0.8
    WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)
    STEP_SAMPLES = int(SAMPLE_RATE * STEP_SECONDS)

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "service": "meeting-ai-stt",
            "model": "nvidia/parakeet-tdt-0.6b-v3",
            "gpu": "A10G",
        }

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        await ws.accept()

        await ws.send_json({
            "type": "connected",
            "message": "Live STT ready",
            "model": "nvidia/parakeet-tdt-0.6b-v3",
        })

        pcm_buffer = bytearray()

        try:
            while True:
                audio_bytes = await ws.receive_bytes()

                if not audio_bytes:
                    continue

                pcm_buffer.extend(audio_bytes)

                print(
                    f"🎙️ PCM RECEIVED: {len(audio_bytes)} bytes "
                    f"| BUFFER: {len(pcm_buffer)} bytes"
                )

                # PCM16 = 2 bytes/sample.
                window_bytes = WINDOW_SAMPLES * 2
                step_bytes = STEP_SAMPLES * 2

                while len(pcm_buffer) >= window_bytes:
                    chunk_bytes = bytes(
                        pcm_buffer[:window_bytes]
                    )

                    # Keep the tail of the current window as overlap for
                    # the next inference pass.
                    del pcm_buffer[:step_bytes]

                    try:
                        audio, sample_rate = (
                            decode_pcm16_to_float32(
                                chunk_bytes
                            )
                        )

                        result = speech_pipe({
                            "raw": audio,
                            "sampling_rate": sample_rate,
                        })

                        text = (
                            result.get("text", "")
                            .strip()
                        )

                        if text:
                            print(
                                f"📝 Transcript: {text}"
                            )

                        await ws.send_json({
                            "type": "transcript",
                            "text": text,
                            "window_seconds": WINDOW_SECONDS,
                            "step_seconds": STEP_SECONDS,
                        })

                    except Exception as exc:
                        print(
                            "STT BUFFER ERROR:",
                            repr(exc)
                        )

                        await ws.send_json({
                            "type": "error",
                            "message": str(exc),
                        })

        except WebSocketDisconnect:
            print(
                "🔌 STT WebSocket disconnected"
            )

        except Exception as exc:
            print(
                "🔴 STT WebSocket error:",
                repr(exc)
            )

    return app

import asyncio
import json
import os
from urllib.parse import urlencode

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from websockets.asyncio.client import connect

SAMPLE_RATE = 16000
DEEPGRAM_URL = "wss://api.deepgram.com/v1/listen?" + urlencode(
    {
        "model": "nova-3",
        "language": "en",
        "encoding": "linear16",
        "sample_rate": SAMPLE_RATE,
        "channels": 1,
        "interim_results": "true",
        "smart_format": "true",
        "punctuate": "true",
        "endpointing": 300,
        "utterance_end_ms": 1000,
        "vad_events": "true",
        "diarize": "true",
    }
)

app = FastAPI(title="Meeting AI Streaming STT")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "meeting-ai-streaming-stt",
        "provider": "deepgram",
        "model": "nova-3",
        "streaming": True,
        "api_key_configured": bool(os.getenv("DEEPGRAM_API_KEY")),
    }


@app.websocket("/ws")
async def websocket_endpoint(client: WebSocket):
    await client.accept()
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        await client.send_json({"type": "error", "message": "STT service is not configured"})
        await client.close(code=1011)
        return

    try:
        async with connect(
            DEEPGRAM_URL,
            additional_headers={"Authorization": f"Token {api_key}"},
            ping_interval=20,
            ping_timeout=20,
            max_size=None,
        ) as deepgram:
            await client.send_json(
                {"type": "connected", "message": "Streaming STT ready", "model": "nova-3"}
            )

            async def forward_audio():
                try:
                    while True:
                        await deepgram.send(await client.receive_bytes())
                except WebSocketDisconnect:
                    try:
                        await deepgram.send(json.dumps({"type": "CloseStream"}))
                    except Exception:
                        pass

            async def forward_transcripts():
                async for raw_message in deepgram:
                    message = json.loads(raw_message)
                    if message.get("type") != "Results":
                        continue
                    alternatives = message.get("channel", {}).get("alternatives", [])
                    if not alternatives:
                        continue
                    transcript = alternatives[0].get("transcript", "").strip()
                    if not transcript:
                        continue
                    words = alternatives[0].get("words", [])
                    segments = []
                    for word in words:
                        speaker = word.get("speaker")
                        token = word.get("punctuated_word") or word.get("word", "")
                        if not token:
                            continue
                        if not segments or segments[-1]["speaker"] != speaker:
                            segments.append({"speaker": speaker, "words": [token]})
                        else:
                            segments[-1]["words"].append(token)
                    if not segments:
                        segments = [{"speaker": None, "words": [transcript]}]
                    for segment in segments:
                        await client.send_json(
                            {
                                "type": "transcript",
                                "text": " ".join(segment["words"]),
                                "is_final": bool(message.get("is_final")),
                                "speech_final": bool(message.get("speech_final")),
                                "start": message.get("start"),
                                "duration": message.get("duration"),
                                "speaker": segment["speaker"],
                            }
                        )

            audio_task = asyncio.create_task(forward_audio())
            transcript_task = asyncio.create_task(forward_transcripts())
            done, pending = await asyncio.wait(
                {audio_task, transcript_task}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                if not task.cancelled():
                    task.result()
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await client.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass

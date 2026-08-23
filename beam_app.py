from beam import Image, asgi


@asgi(
    name="meeting-ai-stt-deepgram",
    cpu=1.0,
    memory=1024,
    image=Image(
        python_version="python3.11",
        python_packages=[
            "fastapi==0.115.12",
            "websockets==15.0.1",
        ],
    ),
    secrets=["DEEPGRAM_API_KEY"],
    authorized=False,
    keep_warm_seconds=10,
    concurrent_requests=5,
    timeout=-1,
)
def web_server(context):
    from server import app

    return app

from beam import Image, QueueDepthAutoscaler, asgi


@asgi(
    name="meeting-ai-stt-deepgram",
    cpu=0.25,
    memory="512Mi",
    image=Image(
        python_version="python3.11",
        python_packages=["fastapi", "websockets"],
    ),
    secrets=["DEEPGRAM_API_KEY"],
    authorized=False,
    keep_warm_seconds=60,
    concurrent_requests=20,
    autoscaler=QueueDepthAutoscaler(
        min_containers=0,
        max_containers=1,
        tasks_per_container=20,
    ),
)
def web_server(context):
    from server import app

    return app

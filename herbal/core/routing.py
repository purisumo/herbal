# your_app/routing.py
from django.urls import path
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
from core.consumers import TrainingProgressConsumer

application = ProtocolTypeRouter({
    "websocket": AuthMiddlewareStack(
        URLRouter(
            # Add your WebSocket consumers here
            path('ws/training_progress/', TrainingProgressConsumer.as_asgi()),
        )
    ),
})

import json
from channels.generic.websocket import AsyncWebsocketConsumer

class ProgressConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

    async def update_progress(self, event):
        progress = event['progress']
        await self.send(text_data=json.dumps({'progress': progress}))

class TrainingProgressConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data):
        data = json.loads(text_data)
        progress = data.get('progress', 0)

        # Send the progress to the WebSocket
        await self.send(text_data=json.dumps({'progress': progress}))

    async def send_progress(self, event):
        progress = event['progress']

        # Send the progress to the WebSocket
        await self.send(text_data=json.dumps({'progress': progress}))
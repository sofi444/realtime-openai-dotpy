import asyncio
import websockets
import json
import pyaudio
import wave
import base64
import logging
import os
import ssl
import threading

from dotenv import load_dotenv

load_dotenv()


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Audio configuration
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 24000

# WebSocket configuration
WS_URL = "wss://api.openai.com/v1/realtime"
MODEL = "gpt-4o-realtime-preview"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

"""
This important message is lost with all log information:
Enter 't' for text, 'a' for audio, or 'q' to quit:
"""

class RealtimeClient:
    def __init__(self):
        logger.info("Initializing RealtimeClient")
        self.ws = None
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.audio_buffer = b''
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE

    async def connect(self):
        logger.info(f"Connecting to WebSocket: {WS_URL}")
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1"
        }
        
        # Check for websockets version and adjust accordingly
        ws_version = tuple(int(x) for x in websockets.__version__.split("."))
        if ws_version < (11, 0):
            # For versions < 11.0, use extra_headers
            self.ws = await websockets.connect(
                f"{WS_URL}?model={MODEL}", 
                extra_headers=headers, 
                ssl=self.ssl_context
            )
        else:
            # For versions >= 11.0, use headers
            self.ws = await websockets.connect(
                f"{WS_URL}?model={MODEL}", 
                headers=headers, 
                ssl=self.ssl_context
            )
        logger.info("Successfully connected to OpenAI Realtime API")

    async def send_event(self, event):
        logger.debug(f"Sending event: {event}")
        await self.ws.send(json.dumps(event))
        logger.debug("Event sent successfully")

    async def receive_events(self):
        try:
            async for message in self.ws:
                logger.debug(f"Received raw message: {message}")
                event = json.loads(message)
                await self.handle_event(event)
        except websockets.ConnectionClosed as e:
            logger.error(f"WebSocket connection closed: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")

    async def handle_event(self, event):
        event_type = event.get("type")
        logger.info(f"Handling event of type: {event_type}")

        if event_type == "error":
            logger.error(f"Error event received: {event['error']['message']}")
        elif event_type == "response.text.delta":
            logger.debug(f"Text delta received: {event['delta']}")
            print(event["delta"], end="", flush=True)
        elif event_type == "response.audio.delta":
            logger.debug(f"Audio delta received, length: {len(event['delta'])}")
            audio_data = base64.b64decode(event["delta"])
            self.audio_buffer += audio_data
        elif event_type == "response.audio.done":
            logger.info("Audio response complete, playing audio")
            self.play_audio(self.audio_buffer)
            self.audio_buffer = b''
        else:
            logger.info(f"Received other event type: {event_type}")

    def start_audio_stream(self):
        logger.info("Starting audio input stream")
        self.stream = self.p.open(format=FORMAT,
                                  channels=CHANNELS,
                                  rate=RATE,
                                  input=True,
                                  frames_per_buffer=CHUNK)
        logger.debug("Audio input stream started successfully")

    def stop_audio_stream(self):
        logger.info("Stopping audio input stream")
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        logger.debug("Audio input stream stopped successfully")

    def record_audio(self, duration):
        logger.info(f"Recording audio for {duration} seconds")
        frames = []
        for i in range(0, int(RATE / CHUNK * duration)):
            data = self.stream.read(CHUNK)
            frames.append(data)
            if i % 10 == 0:  # Log every 10th frame
                logger.debug(f"Recorded frame {i}")
        audio_data = b''.join(frames)
        logger.info(f"Audio recording complete, total size: {len(audio_data)} bytes")
        return audio_data

    def play_audio(self, audio_data):
        logger.info(f"Playing audio, size: {len(audio_data)} bytes")

        def play():
            stream = self.p.open(format=FORMAT,
                                channels=CHANNELS,
                                rate=RATE,
                                output=True)
            stream.write(audio_data)
            stream.stop_stream()
            stream.close()
            logger.debug("Audio playback complete")

        playback_thread = threading.Thread(target=play)
        playback_thread.start()

    async def send_text(self, text):
        logger.info(f"Sending text message: {text}")
        event = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{
                    "type": "input_text",
                    "text": text
                }]
            }
        }
        await self.send_event(event)
        logger.debug("Text message sent, creating response")
        await self.send_event({"type": "response.create"})

    async def send_audio(self, duration):
        logger.info(f"Preparing to send audio of duration: {duration} seconds")
        self.start_audio_stream()
        audio_data = self.record_audio(duration)
        self.stop_audio_stream()

        base64_audio = base64.b64encode(audio_data).decode('utf-8')
        logger.debug(f"Audio encoded to base64, length: {len(base64_audio)}")
        
        event = {
            "type": "input_audio_buffer.append",
            "audio": base64_audio
        }
        await self.send_event(event)
        logger.debug("Audio buffer appended, committing buffer")
        await self.send_event({"type": "input_audio_buffer.commit"})
        logger.debug("Audio buffer committed, creating response")
        await self.send_event({"type": "response.create"})

    async def run(self):
        await self.connect()
        
        # Continuously listen to events in the background
        receive_task = asyncio.create_task(self.receive_events())
        
        try:
            while True:
                command = await asyncio.get_event_loop().run_in_executor(None, input, "Enter 't' for text, 'a' for audio, or 'q' to quit: ")
                if command == 'q':
                    logger.info("Quit command received")
                    break
                elif command == 't':
                    text = await asyncio.get_event_loop().run_in_executor(None, input, "Enter your message: ")
                    await self.send_text(text)
                elif command == 'a':
                    await self.send_audio(5)
                await asyncio.sleep(0.1)
        except Exception as e:
            logger.error(f"An error occurred: {e}")
        finally:
            receive_task.cancel()
            await self.cleanup()

    async def cleanup(self):
        logger.info("Cleaning up resources")
        if self.stream:
            self.stop_audio_stream()
        if self.ws:
            await self.ws.close()


async def main():
    logger.info("Starting main function")
    client = RealtimeClient()
    try:
        await client.run()
    except Exception as e:
        logger.error(f"An error occurred in main: {e}")
    finally:
        logger.info("Main function completed")

if __name__ == "__main__":
    logger.info("Script started")
    asyncio.run(main())
    logger.info("Script completed")
import asyncio
import websockets
import json
import pyaudio
import base64
import logging
import os
import ssl
import threading

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TOPIC = "PARIS"
INSTRUCTIONS = f"""
You are Jojo, an experienced, friendly English teacher. Have a conversation with your student about {TOPIC}.
Adapt your speech to their proficiency level. If you detect that the student is a beginner, speak slowly enunciating every word properly, use vocabulary and sentence structure appropriate for beginners. If you detect that the student is more advanced, you can gradually introduce more complex vocabulary, idiomatic expressions and sentence structure. At all times, ensure the student can follow the conversation and is comfortable.
Only speak one sentence or question per turn.
When you see the prompt 'START', you start the conversation.
"""

class AudioHandler:
    """
    Handles audio input and output using PyAudio.
    """
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.audio_buffer = b''
        self.chunk_size = 1024  # Number of audio frames per buffer
        self.format = pyaudio.paInt16  # Audio format (16-bit PCM)
        self.channels = 1  # Mono audio
        self.rate = 24000  # Sampling rate in Hz
        self.is_recording = False

    def start_audio_stream(self):
        """
        Start the audio input stream.
        """
        self.stream = self.p.open(format=self.format,
                                  channels=self.channels,
                                  rate=self.rate,
                                  input=True,
                                  frames_per_buffer=self.chunk_size)

    def stop_audio_stream(self):
        """
        Stop the audio input stream.
        """
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()

    def cleanup(self):
        """
        Clean up resources by stopping the stream and terminating PyAudio.
        """
        if self.stream:
            self.stop_audio_stream()
        self.p.terminate()

    def start_recording(self):
        """Start continuous recording"""
        self.is_recording = True
        self.audio_buffer = b''
        self.start_audio_stream()

    def stop_recording(self):
        """Stop recording and return the recorded audio"""
        self.is_recording = False
        self.stop_audio_stream()
        return self.audio_buffer

    def record_chunk(self):
        """Record a single chunk of audio"""
        if self.stream and self.is_recording:
            data = self.stream.read(self.chunk_size)
            self.audio_buffer += data
            return data
        return None
    
    def play_audio(self, audio_data):
        """
        Play audio data.
        
        :param audio_data: Audio data to play.
        """
        def play():
            stream = self.p.open(format=self.format,
                                 channels=self.channels,
                                 rate=self.rate,
                                 output=True)
            stream.write(audio_data)
            stream.stop_stream()
            stream.close()

        # Use a separate thread for playback to avoid blocking
        playback_thread = threading.Thread(target=play)
        playback_thread.start()


class RealtimeClient:
    """
    Client for interacting with the OpenAI Realtime API via WebSocket.
    """
    def __init__(self, instructions, voice="alloy"):
        # WebSocket Configuration
        self.url = "wss://api.openai.com/v1/realtime"  # WebSocket URL for OpenAI API
        self.model = "gpt-4o-realtime-preview"  # Model identifier
        self.api_key = os.getenv("OPENAI_API_KEY")  # API key from environment variable
        self.ws = None
        self.audio_handler = AudioHandler()
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE
        self.audio_buffer = b''  # Buffer for streaming audio responses
        self.instructions = instructions
        self.voice = voice

        # VAD mode
        self.turn_detection_config = {
            "type": "server_vad",
            "threshold": 0.5,
            "prefix_padding_ms": 100,
            "silence_duration_ms": 600
        }

    async def connect(self):
        """
        Connect to the WebSocket server.
        """
        logger.info(f"Connecting to WebSocket: {self.url}")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1"
        }
        
        # NEEDS websockets version < 11.0
        self.ws = await websockets.connect(
            f"{self.url}?model={self.model}",
            extra_headers=headers,
            ssl=self.ssl_context
        )
        logger.info("Successfully connected to OpenAI Realtime API")

        # Set up session configuration
        await self.update_session({
            "modalities": ["audio", "text"],
            "instructions": self.instructions,
            "voice": self.voice,
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "turn_detection": self.turn_detection_config
        })
        logger.info("Session updated.")
        logger.info(f"Instructions: {self.instructions}")
        logger.info(f"Voice: {self.voice}")

    async def update_session(self, config):
        """
        Update the session configuration.
        """
        event = {
            "type": "session.update",
            "session": config
        }
        await self.send_event(event)

    async def send_event(self, event):
        """
        Send an event to the WebSocket server.
        
        :param event: Event data to send.
        """
        logger.debug(f"Sending event type: {event['type']}")
        await self.ws.send(json.dumps(event))
        logger.debug("Event sent successfully")

    async def receive_events(self):
        """
        Continuously receive events from the WebSocket server.
        """
        try:
            async for message in self.ws:
                event = json.loads(message)
                await self.handle_event(event)
        except websockets.ConnectionClosed as e:
            logger.error(f"WebSocket connection closed: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")

    async def handle_event(self, event):
        """
        Handle incoming events from the WebSocket server.
        
        :param event: Event data received.
        """
        event_type = event.get("type")
        logger.debug(f"Received event type: {event_type}")

        if event_type == "error":
            logger.error(f"Error event received: {event['error']['message']}")
        elif event_type == "response.text.delta":
            # Print text response incrementally
            print(event["delta"], end="", flush=True)
        elif event_type == "response.audio.delta":
            # Append audio data to buffer
            audio_data = base64.b64decode(event["delta"])
            self.audio_buffer += audio_data
            logger.debug("Audio data appended to buffer")
        elif event_type == "response.audio.done":
            # Play the complete audio response
            if self.audio_buffer:
                self.audio_handler.play_audio(self.audio_buffer)
                logger.info("Playing audio response")
                self.audio_buffer = b''
            else:
                logger.warning("No audio data to play")

    async def send_text(self, text):
        """
        Send a text message to the WebSocket server.
        
        :param text: Text message to send.
        """
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

    async def send_audio(self):
        """
        Record and send audio using server-side turn detection
        """
        self.audio_handler.start_recording()
        
        try:
            while True:
                chunk = self.audio_handler.record_chunk()
                if chunk:
                    # Encode and send audio chunk
                    base64_chunk = base64.b64encode(chunk).decode('utf-8')
                    await self.send_event({
                        "type": "input_audio_buffer.append",
                        "audio": base64_chunk
                    })
                    await asyncio.sleep(0.01)
                else:
                    break

        except Exception as e:
            logger.error(f"Error during audio recording: {e}")
            self.audio_handler.stop_recording()
        
        await self.send_event({"type": "input_audio_buffer.commit"})
        await self.send_event({"type": "response.create"})

    async def run(self):
        """
        Main loop to handle user input and interact with the WebSocket server.
        """
        await self.connect()
        
        # Continuously listen to events in the background
        receive_task = asyncio.create_task(self.receive_events())
        
        # Fake first user message (not played)
        await self.send_text("START")

        try:
            while True:
                # Get user command input
                command = await asyncio.get_event_loop().run_in_executor(None, input, "Enter 't' for text, 'a' for audio, or 'q' to quit: ")
                if command == 'q':
                    logger.info("Quit command received")
                    break
                elif command == 't':
                    # Get text input from user
                    text = await asyncio.get_event_loop().run_in_executor(None, input, "Enter your message: ")
                    await self.send_text(text)
                elif command == 'a':
                    # Record and send audio
                    await self.send_audio()
                await asyncio.sleep(0.1)
        except Exception as e:
            logger.error(f"An error occurred: {e}")
        finally:
            receive_task.cancel()
            await self.cleanup()

    async def cleanup(self):
        """
        Clean up resources by closing the WebSocket and audio handler.
        """
        self.audio_handler.cleanup()
        if self.ws:
            await self.ws.close()

async def main():

    client = RealtimeClient(
        instructions=INSTRUCTIONS,
        voice="alloy"
    )
    try:
        await client.run()
    except Exception as e:
        logger.error(f"An error occurred in main: {e}")
    finally:
        logger.info("Main function completed")

if __name__ == "__main__":
    asyncio.run(main())
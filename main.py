import asyncio
import traceback
import pyaudio
import numpy as np
from google import genai
from google.genai import types

# === CONFIGURATION ===
GEMINI_API_KEY = "YOUR_API_KEY_HERE"  # <--- PASTE KEY HERE
MODEL = "models/gemini-2.5-flash-native-audio-preview-12-2025"

# The "Persona" - This tells the AI how to behave
CUSTOM_PROMPT = (
    "Your name is SinhalaBot. You are a helpful AI assistant. "
    "Answer in a mix of Sinhala and English (Singlish). "
    "Keep responses short, witty, and fun. Do not mention you are an AI."
)

# === AUDIO SETTINGS (Tuned for Pi 4) ===
FORMAT = pyaudio.paInt16
CHANNELS = 1
HARDWARE_IN_RATE = 48000   # Standard USB Mic rate
HARDWARE_OUT_RATE = 24000  # Gemini Native Rate (Works on most Pi jacks)
CHUNK = 1024               # Buffer size

client = genai.Client(http_options={"api_version": "v1beta"}, api_key=GEMINI_API_KEY)
CONFIG = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Puck")
        )
    ),
)

pya = pyaudio.PyAudio()

class SinhalaBot:
    def __init__(self):
        self.audio_in_queue = asyncio.Queue()
        self.out_queue = asyncio.Queue(maxsize=5)
        self.session = None
        self.should_exit = False

    async def listen_mic(self):
        """Captures audio from USB Mic and downsamples it for the AI"""
        stream = pya.open(format=FORMAT, channels=CHANNELS, rate=HARDWARE_IN_RATE, 
                         input=True, frames_per_buffer=CHUNK)
        print("Listening...")
        while not self.should_exit:
            data = await asyncio.to_thread(stream.read, CHUNK, exception_on_overflow=False)
            # Simple downsampling trick for Pi CPU (48k -> 16k)
            audio_array = np.frombuffer(data, dtype=np.int16)
            downsampled_data = audio_array[::3].tobytes()
            await self.out_queue.put({"mime_type": "audio/pcm", "data": downsampled_data})

    async def play_speaker(self):
        """Plays audio received from AI directly to speakers"""
        stream = pya.open(format=FORMAT, channels=CHANNELS, rate=HARDWARE_OUT_RATE, output=True)
        while not self.should_exit:
            data = await self.audio_in_queue.get()
            await asyncio.to_thread(stream.write, data)

    async def run(self):
        async with client.aio.live.connect(model=MODEL, config=CONFIG) as session:
            self.session = session
            # Send the persona instruction first
            await self.session.send(input=CUSTOM_PROMPT, end_of_turn=True)
            print("SinhalaBot is LIVE! Say something in Sinhala...")
            
            # Run all tasks simultaneously
            await asyncio.gather(
                self.listen_mic(),
                self.play_speaker(),
                self.receive_loop(),
                self.send_loop()
            )

    async def send_loop(self):
        while not self.should_exit:
            msg = await self.out_queue.get()
            await self.session.send(input=msg)

    async def receive_loop(self):
        while not self.should_exit:
            async for response in self.session.receive():
                if response.data: await self.audio_in_queue.put(response.data)

if __name__ == "__main__":
    bot = SinhalaBot()
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        print("\n SinhalaBot signing off!")
    finally:
        pya.terminate()

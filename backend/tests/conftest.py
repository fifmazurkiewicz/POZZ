import os

os.environ.pop("DATABASE_URL", None)
os.environ["VOICE_MODE"] = "speech_to_speech"
os.environ["TTS_PROVIDER"] = "elevenlabs"

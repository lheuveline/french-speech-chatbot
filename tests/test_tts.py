import subprocess
import sys
sys.path.append("../")
from src.chatbot.clients import TTSClient

client = TTSClient("10.5.0.3", "5000")

tts_endpoint = "http://10.5.0.3:5000/generate"

text = "Ceci est un test."
audio_bytes = client.make_request(text)

with open("tts_test_output.wav", "wb") as f:
    f.write(audio_bytes)

cmd = ["mediainfo", "tests/tts_test_output.wav"]
mediainfo_output = subprocess.check_output(cmd).decode()
print(mediainfo_output)
import numpy as np
from scipy.io.wavfile import read
import torch

import sys
sys.path.append("../")
from src.chatbot.clients import ChatbotClient

client = ChatbotClient()
client.debug = True
client.psychotic_mode = True

client.llm_client.llm_endpoint = "http://10.5.0.2:8000/generate"
client.tts_client.tts_endpoint = "http://10.5.0.3:5000/generate"
client.asr_client.asr_endpoint = "http://10.5.0.4:5000/transcribe"

fake_input_filename = "tests/asr_test_input.wav"

wav_filename = "tests/asr_test_input.wav"
sr, audio_data = read(wav_filename)
audio_data = audio_data.astype(np.float32) / 32768.0
audio_data = torch.Tensor(audio_data)

audio_data = {"audio_data" : audio_data}
#client.process_mic_input(audio_data)

client.run()

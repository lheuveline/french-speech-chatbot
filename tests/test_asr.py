import requests
import torch
import json

import numpy as np

from scipy.io.wavfile import read

ASR_ENDPOINT="http://10.5.0.4:5000/transcribe"

wav_filename = "tests/asr_test_wav.wav"
sr, audio_data = read(wav_filename)
audio_data = audio_data.astype(np.float32) / 32768.0
audio_data = audio_data.tolist()

headers = {
    "Content-Type": "application/json"
}
query = {
    "audio_data" : audio_data
}

r = requests.post(
    ASR_ENDPOINT, data=json.dumps(query), headers=headers
)
print(r)
if r.status_code != 200:
    print(r.text)
else: 
    print(r.json())
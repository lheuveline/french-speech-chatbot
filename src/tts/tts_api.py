import io
import os

from flask import Flask, request, send_file
from scipy.io.wavfile import write

import numpy as np
import torch
from TTS.api import TTS

DEFAULT_MODEL="tts_models/fr/mai/tacotron2-DDC"

class TTSClient:

    def __init__(
            self,
            model=None
        ):
        
        if not model:
            self.model = os.environ.get("MODEL", DEFAULT_MODEL)
        else:
            self.model = model

        # Get device
        device = os.environ.get("DEVICE", "cpu")
        # Init TTS
        self.tts = TTS(self.model).to(device)
        self.tts_sample_rate = 22050

    def write_bytes(self, wav):

        file_obj = io.BytesIO()
        write(file_obj, self.tts_sample_rate, wav)
        file_obj.seek(0)
        return file_obj

    def inference(self, text):

        # Run TTS
        wav = np.asarray(self.tts.tts(text=text))
        wav_bytes = self.write_bytes(wav)
        return wav_bytes

print("Loading TTS model...")
tts_client = TTSClient()
print("Starting API server...")
app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def hello_world():
    request_data = request.get_json()

    text = request_data["text"]
    wav_bytes = tts_client.inference(text)

    return send_file(
        wav_bytes,
        mimetype="audio/wav"
    )
    
if __name__ == '__main__':
    # run app in debug mode on port 5000
    app.run(debug=True, host="0.0.0.0", port=5000)

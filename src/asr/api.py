import whisper
from flask import Flask, request, send_file
import os
import torch

print("Starting API server...")
app = Flask(__name__)
print("Loading model...")
model_name = os.environ.get("MODEL", "base")
model = whisper.load_model(model_name)
language = os.environ.get("LANGUAGE", "en")

@app.route('/transcribe', methods=['POST'])
def transcribe():
    request_data = request.get_json()
    audio_data = torch.Tensor(request_data["audio_data"])
    result = model.transcribe(audio_data, language=language)
    return result["text"]

if __name__ == '__main__':
    # run app in debug mode on port 5000
    app.run(debug=True, port=5000)

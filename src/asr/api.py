from flask import Flask, request, send_file
import os
import numpy as np
import json

from transformers import pipeline

print("Starting API server...")
app = Flask(__name__)
print("Loading model...")
model = os.environ.get("MODEL", "base")
device = os.environ.get("DEVICE")

pipe = pipeline(
    "automatic-speech-recognition", 
    model=model, 
    device=device
)

@app.route('/transcribe', methods=['POST'])
def transcribe():
    request_data = request.get_json()
    audio_data = np.asarray(request_data["audio_data"])
    result = pipe(audio_data, max_new_tokens=225)["text"]
    return json.dumps(result)

if __name__ == '__main__':
    # run app in debug mode on port 5000
    app.run(debug=True, host="0.0.0.0", port=5000)

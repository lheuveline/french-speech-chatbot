import json

from optimum.onnxruntime import ORTModelForCTC
from transformers import Wav2Vec2Processor
import numpy as np
import torch

from flask import Flask, request, send_file

MODEL_ID = "Poulpidot/wav2vec2-large-xlsr-53-french-onnx"

print("Starting API server...")
app = Flask(__name__)
print("Loading model...")
model = ORTModelForCTC.from_pretrained(MODEL_ID)
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)

def predict(wav):
    inputs = processor(
        wav, 
        sampling_rate=16_000, 
        return_tensors="pt", 
        padding=True
    )

    with torch.no_grad():
        logits = model(
            inputs.input_values, attention_mask=inputs.attention_mask).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    predicted_sentences = processor.batch_decode(predicted_ids)
    predicted_sentences = " .".join(predicted_sentences)
    return predicted_sentences

@app.route('/transcribe', methods=['POST'])
def transcribe():
    request_data = request.get_json()
    audio_data = np.asarray(request_data["audio_data"])
    result = predict(audio_data)
    return json.dumps(result)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)

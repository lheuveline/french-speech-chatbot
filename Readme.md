# French Speech Chatbot

## Presentation

This project is a deployment stack for a full speech capable chatbot in french language.
It uses Docker-Compose to run separate API's for each task: ASR, Language generation, TTS.
Using docker-compose architecture, all models can run on remote host, allowing to run chatbot app on low performance hardware. This way, the compute cost can be cut if running on cloud by starting the stack only when used.
For windows users, chatbot must be run using `python src/chatbot/clients.py`. Linux users can use docker if ensuring usb device sharing between host and container.

The Chatbot uses microphone input to detect speech. If its wake_up word ("Alfred" by default) is detected, it will treat the spoken sentence as a request.
When the Chatbot detect a request, it will send request to the LLM to get an answer and then use TTS to synthesize response and play the output through speakers.

- TO-DO :
  * [WIP] Add a way to use local models each service
  * [NotStarted] Add service to enable LLM model finetuning
  * [Done !] Reduce ASR model GPU usage
  * [NotStarted] Get or train a better French TTS model

## Requirements

Running the project requires at least 14GB of VRAM.

## Usage

* Start the Chatbot in the background :

``` docker compose -f docker/docker-compose.yaml up -d ```

Start talking using `Alfred` wake up word and wait for a response.

* Environment variables :

vLLM: Server running LLM model
```
MODEL : Model to load for vLLM server
QUANTIZATION : Quantization mode
MAX_MODEL_LEN : maximum model length
GPU_MEMORY_UTILIZATION : GPU utilization ratio
```

TTS: Custom CoquiTTS server running TTS model
```
MODEL : TTS model to load with CoquiTTS
DEVICE : Device to use for TTS model : "cpu" or "cuda"
```

ASR: Custom API serving wav2vec2 french onnx model predictions
```
MODEL : Transformers ASR model to use
DEVICE : Device to use for ASR model : "cpu" or "cuda"
```

Chatbot: Main app entrypoint implementing chatbot logic : Microphone listening, wake up word, model api's requests
```
NAME : Chatbot's name, used as wake up word.
AUTONOMOUS_MODE : If set to `true`, the chatbot will talk to himself. No ASR.
```

## Aknowledgements

* vLLM
* CoquiTTS
* Whisper-Mic

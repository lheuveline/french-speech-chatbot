# French Speech Chatbot

## Presentation

This project is a deployment stack for a full speech capable chatbot in french language.
It uses Docker-Compose to run separate API's for each task: ASR, Language generation, TTS.

The Chatbot uses microphone input to detect speech. If its wake_up word ("Alfred") is detected, it will treat the spoken sentence as a request.
When the Chatbot detect a request, it will send request to the LLM to get an answer and then use TTS to synthesize response and play the output through speakers.

- TO-DO :
  * Add a way to use local models each service
  * Add service to enable LLM model finetuning
  * Reduce ASR model GPU usage
  * Get or train a better French TTS model

## Requirements

Running the project requires at least 14GB of VRAM.

## Usage

* Start the Chatbot in the background :

``` docker compose -f docker/docker-compose.yaml up -d ```

Start talking using `Alfred` wake up word and wait for a response.

* Environment variables :

vLLM:
```
MODEL : Model to load for vLLM server
QUANTIZATION : Quantization mode
MAX_MODEL_LEN : maximum model length
GPU_MEMORY_UTILIZATION : GPU utilization ratio
```

TTS:
```
MODEL : TTS model to load with CoquiTTS
DEVICE : Device to use for TTS model : "cpu" or "cuda"
```

ASR:
```
MODEL : Transformers ASR model to use
DEVICE : Device to use for ASR model : "cpu" or "cuda"
```

Chatbot:
```
NAME : Chatbot's name, used as wake up word.
AUTONOMOUS_MODE : If set to `true`, the chatbot will talk to himself. No ASR.
```

## Aknowledgements

* vLLM
* CoquiTTS
* Whisper
* Whisper-Mic

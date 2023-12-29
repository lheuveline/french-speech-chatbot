#### French Speech Chatbot

# Presentation

This project is a deployment stack for a full speech capable chatbot in french language.
It uses Docker-Compose to run separate API's for each task: ASR, Language generation, TTS.

The Chatbot uses microphone input to detect speech. If its wake_up word ("Alfred") is detected, it will treat the spoken sentence as a request.
When the Chatbot detect a request, it will send request to the LLM to get an answer and then use TTS to synthesize response and play the output through speakers.

- TO-DO :
  * Add a way to use local models for LLM server
  * Add service to enable LLM model finetuning
  * Reduce ASR model GPU usage
  * Get or train a better French TTS model

# Requirements

Running the project requires at least 14GB of VRAM.

# Usage

* Start the Chatbot in the background :
``` docker compose -f docker/docker-compose.yaml up -d ```

Start talking using `Alfred` wake up word and wait for a response.

# Aknowledgements

* vLLM
* CoquiTTS
* Whisper
* Whisper-Mic

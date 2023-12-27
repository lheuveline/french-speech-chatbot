import requests
import json
import os

from .WhisperMic import WhisperMic

class LLMClient:

    def __init__(
        self,
        llm_api_host=None,
        llm_api_port=None,
        system_message=None
    ):
        
        if not llm_api_host:
            self.llm_api_host = os.environ.get("LLM_API_HOST","localhost")
        else:
            self.llm_api_host = llm_api_host
        if not llm_api_port:
            self.llm_api_port = os.environ.get("LLM_API_PORT", "8000")
        else:
            self.llm_api_port = llm_api_port

        self.llm_endpoint = f"http://{self.llm_api_host}:{self.llm_api_port}/generate"

        if not system_message:
            self.system_message = """
                Vous êtes un assistant français chargé de répondre le plus 
                justement possible. 
                """.replace("\n", "")

        self.default_params = {
            "temperature" : 0.2, 
            "top_p" : 0.95,
            "repetition_penalty" : 1.5,
            "max_tokens"  : 1024,
            "stop" : "."
        }

    def set_template(self, text):

        return f'''<s>[INST] <<SYS>>
            {self.system_message}
            <</SYS>>
            
            {text} [/INST] 
            '''

    def parse_response(self, response):

        parsed_response = response.json()["text"][0] \
            .split('[/INST]')[-1] \
            .replace("\n", "") \
            .strip()
        return parsed_response

    def make_request(self, text:str, params:dict = None):

        if not params:
            params = self.default_params
        
        data = {
            "prompt" : self.set_template(text)
        }
        data.update(params)
        
        response = requests.post(self.llm_endpoint, data=json.dumps(data))
        parsed_response = self.parse_response(response)
        return parsed_response

class TTSClient:

    def __init__(
        self,
        tts_api_host=None,
        tts_api_port=None
    ): 
        
        if tts_api_host:
            self.tts_api_host = tts_api_host
        else:
            self.tts_api_host = os.environ.get("TTS_API_HOST", "localhost")

        if tts_api_port:
            self.tts_api_port = tts_api_port
        else:
            self.tts_api_port = os.environ.get("TTS_API_PORT", "8000")

        self.tts_endpoint = f"http://{self.tts_api_host}:{self.tts_api_port}/generate"

    def parse_response(self, response):

        return response.content

    def make_request(self, text):

        data = {"text", text}
        response = requests.post(self.tts_endpoint, data=json.dumps(data))
        audio_bytes = self.parse_response(response)
        return audio_bytes

class ASRClient:

    def __init__(
        self,
        model=None
    ):
        
        if not model:
            self.model = os.environ.get("MODEL", "base")
        else:
            self.model = model

        self.phrase_time_limit = 10

        self.mic = WhisperMic(
            model=model
        )

    def run(self):
        self.mic.listen_loop(phrase_time_limit=self.phrase_time_limit)

class ChatbotClient:

    def __init__(self):

        self.llm_client = LLMClient()
        self.tts_client = TTSClient()
        self.asr_client = ASRClient()

        self.wake_up_word = "Alfred"

    def process_asr_result(self, result):
        
        """Trigger requests if self.wake_up_word is in asr result"""
        
        if self.wake_up_word in result:
            clean_result = result.replace(self.wake_up_word, "", 1)

            print("Getting LLM response...")
            llm_response = self.llm_client.make_request(clean_result)
            print("Getting TTS response...")
            tts_response = self.tts_client.make_request(llm_response)
            return tts_response
        
    def play_tts_response(self, response):

        wav_bytes = response.content

    def run(self):

        """
        - Start ASR listen loop
        - If self.wake_up_word is detected in transcribe:
          - send detected text after self.wake_up_word to LLM API
          - send LLM response to TTS API
          - Play TTS API response (audio_bytes)
        """

        try:
            self.asr_client.run()
            while True:
                result = self.mic.result_queue.get()
                print(result)
                audio_output = self.process_asr_result(result)
                self.play_tts_response(audio_output)
        except KeyboardInterrupt:
            print("Operation interrupted successfully")

if __name__ == "__main__":

    chatbot_client = ChatbotClient()
    chatbot_client.run()
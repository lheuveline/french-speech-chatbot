import requests
import json
import os
import subprocess
import re
import time

from mic_handler import MicClient

class LLMClient:

    def __init__(
        self,
        llm_api_host=None,
        llm_api_port=None,
        system_message=None,
        remove_punctuation=False,
        debug=False
    ):
        
        if not llm_api_host:
            self.llm_api_host = os.environ.get("LLM_API_HOST","localhost")
        else:
            self.llm_api_host = llm_api_host
        if not llm_api_port:
            self.llm_api_port = os.environ.get("LLM_API_PORT", "8000")
        else:
            self.llm_api_port = llm_api_port

        self.remove_punctuation = os.environ.get(
            "REMOVE_PUNCTUATION", 
            remove_punctuation
        )
        self.debug = os.environ.get("DEBUG", False)

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
        
        if self.remove_punctuation:
            for punct in [",.!?"]:
                parsed_response = parsed_response \
                    .replace(punct, "")

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

        data = {"text" : text}
        headers = {"Content-Type" : "application/json"}
        response = requests.post(
            self.tts_endpoint, 
            data=json.dumps(data), 
            headers=headers
        )
        audio_bytes = self.parse_response(response)
        return audio_bytes

class ChatbotClient:

    def __init__(self):

        self.llm_client = LLMClient()
        self.tts_client = TTSClient()
        self.asr_client = MicClient()

        self.wake_up_word = os.environ.get("NAME", "Alfred")
        self.wake_up_word = self.wake_up_word.lower()

        debug = os.environ.get("DEBUG", "False")
        self.debug = True if debug.lower() in ["true", "1"] else False

        autonomous_mode = os.environ.get("AUTONOMOUS_MODE", "False")
        self.autonomous_mode = True if autonomous_mode.lower() in ["true", "1"] else False

        if not self.autonomous_mode:
            self.mic_client = MicClient()

    def process_mic_input(self, result):
        
        """Trigger requests if self.wake_up_word is in asr result"""

        asr_output = self.asr_client.make_request(result["audio_data"]).json()
        print("Listened :", asr_output)
        print(self.wake_up_word in asr_output[0])
        
        if self.wake_up_word in asr_output:
            # Need to cut sentence and keep chars only after self.wake_up_word
            clean_result = asr_output.replace(self.wake_up_word, "", 1)

            print("Getting LLM response...")
            llm_response = self.llm_client.make_request(clean_result)
            print("LLM response:", llm_response)
            print("Getting TTS response...")
            tts_response = self.tts_client.make_request(llm_response)
            self.play_tts_response(tts_response)
        
    def play_tts_response(self, response):
        
        temp_filename = "tmp.wav"
        with open(temp_filename, "wb") as f:
            f.write(response)

        cmd = ["play", temp_filename]
        subprocess.check_output(cmd)
        if not self.debug:
            os.remove(temp_filename)

    def run_autonomous_mode(self):

        """
        Enable psychotic mode.
        In this mode, the chatbot will talk to himself by getting response to LLM responses and synthesizing them back. asr is not used in this mode.
        """

        def parse_llm_response(text):

            to_clean = [
                r"^.*:"
            ]
            clean = text
            for pattern in to_clean:
                clean = re.sub(pattern, "", clean)
            clean = clean.strip()
            return clean

        seed = "Donne moi un exemple de grand évènement historique et pose une question."

        current_text = seed
        try:
            while True:
                # Direct LLM call
                llm_response = self.llm_client.make_request(current_text)
                print("LLM response:", llm_response)
                llm_response = parse_llm_response(llm_response)

                tts_response = self.tts_client.make_request(llm_response)
                self.play_tts_response(tts_response)

                llm_response = llm_response + ". Pose une question."
                current_text = llm_response

                time.sleep(2)

        except KeyboardInterrupt:
            print("Quitting...")

    def run(self):

        if self.autonomous_mode:
            self.run_autonomous_mode()
        else:
            try:
                self.mic_client.listen_loop()
            except KeyboardInterrupt:
                print("Operation interrupted successfully")

if __name__ == "__main__":

    chatbot_client = ChatbotClient()
    chatbot_client.run()
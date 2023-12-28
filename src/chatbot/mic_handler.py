import json
import queue
import numpy as np
import speech_recognition as sr
import time
import threading
import torch
import requests
import os

import logging
from typing_extensions import Literal
from rich.logging import RichHandler

def get_logger(name: str, level: Literal["info", "warning", "debug"]) -> logging.Logger:
    rich_handler = RichHandler(level=logging.INFO, rich_tracebacks=True, markup=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging._nameToLevel[level.upper()])

    if not logger.handlers:
        logger.addHandler(rich_handler)

    logger.propagate = False

    return logger

class MicHandler:

    def __init__(
            self,
            energy=300,
            pause=2,
            dynamic_energy=False,
            mic_index=None
    ): 
        self.logger = get_logger("whisper_mic", "info")
        self.energy = energy
        self.pause = pause
        self.dynamic_energy = dynamic_energy

        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()

        self.__setup_mic(mic_index)


    def __setup_mic(self, mic_index):
        if mic_index is None:
            self.logger.info("No mic index provided, using default")
        self.source = sr.Microphone(sample_rate=16000, device_index=mic_index)

        self.recorder = sr.Recognizer()
        self.recorder.energy_threshold = self.energy
        self.recorder.pause_threshold = self.pause
        self.recorder.dynamic_energy_threshold = self.dynamic_energy

        with self.source:
            self.recorder.adjust_for_ambient_noise(self.source)

        self.logger.info("Mic setup complete")

    def __get_all_audio(self, min_time: float = -1.):
        audio = bytes()
        got_audio = False
        time_start = time.time()
        try:
            while not got_audio or time.time() - time_start < min_time:
                while not self.audio_queue.empty():
                    audio += self.audio_queue.get()
                    got_audio = True
        except KeyboardInterrupt:
            exit(1)

        data = sr.AudioData(audio,16000,2)
        data = data.get_raw_data()
        return data
    
    # This method takes the recorded audio data, converts it into raw format and stores it in a queue. 
    def __record_load(self,_, audio: sr.AudioData) -> None:
        data = audio.get_raw_data()
        self.audio_queue.put_nowait(data)

    def __preprocess(self, data):
        return torch.from_numpy(np.frombuffer(data, np.int16).flatten().astype(np.float32) / 32768.0)
    
    def __listen(self, data=None):
        if data is None:
            audio_data = self.__get_all_audio()
        else:
            audio_data = data
        audio_data = self.__preprocess(audio_data)

        self.result_queue.put_nowait(audio_data)

    def __listen_forever(self) -> None:

        try:
            while True:
                self.__listen()
        except KeyboardInterrupt:
            exit(1)

    def __postprocess(self, audio_data):
        print("Received :", audio_data)

    def listen_loop(self, phrase_time_limit=None) -> None:

        try:
            self.recorder.listen_in_background(
                self.source, 
                self.__record_load, 
                phrase_time_limit=phrase_time_limit
            )
            self.logger.info("Listening...")
            threading.Thread(
                target=self.__listen_forever, daemon=True
            ).start()

            while True:
                result = self.result_queue.get()
                self.__postprocess(result)
        except KeyboardInterrupt:
            exit(1)

class MicClient(MicHandler):

    """
    Take input from microphone, convert raw data to torch Tensor (sample rate = 16000) and send tensor to asr endpoint.
    """

    def __init__(
            self,
            asr_api_host=None,
            asr_api_port=None
        ):
        
        super().__init__()

        if asr_api_host:
            self.asr_api_host = asr_api_host
        else:
            self.asr_api_host = os.environ.get("ASR_API_HOST", "localhost")
        if asr_api_port:
            self.asr_api_port = asr_api_port
        else:
            self.asr_api_port = os.environ.get("ASR_API_PORT", "8000")

        self.asr_endpoint = f"http://{self.asr_api_host}:{self.asr_api_port}/transcribe"

    def __postprocess(self, audio_data):
        self.make_request(audio_data)

    def make_request(self, audio_data:torch.Tensor):

        data = audio_data.cpu().numpy().tolist()
        headers = {
            "Content-Type": "application/json"
        }
        payload = {
            "audio_data" : data
        }
        
        response = requests.post(
            self.asr_endpoint,
            data=json.dumps(payload),
            headers=headers
        )
        return response

if __name__ == "__main__":

    mic_handler = MicHandler()
    try:
        mic_handler.listen_loop()
    except KeyboardInterrupt:
        print("Stopping...")

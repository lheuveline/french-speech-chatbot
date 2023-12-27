import os
import threading

import whisper_mic

class WhisperMic(whisper_mic.WhisperMic):

    def __transcribe(
            self,
            data=None, 
            realtime: bool = False
    ) -> None:
        
        """
        Override transcribe method to accept non-english language
        Set "LANGUAGE" environment variable to enable
        """
        
        if data is None:
            audio_data = self.__get_all_audio()
        else:
            audio_data = data
        audio_data = self.__preprocess(audio_data)

        if self.english:
            result = self.audio_model.transcribe(
                audio_data,
                language='english'
            )
        elif os.environ("LANGUAGE", None) is not None:
            language = os.environ("LANGUAGE")
            result = self.audio_model.transcribe(
                audio_data,
                language=language
            )
        else:
            result = self.audio_model.transcribe(audio_data)

        predicted_text = result["text"]
        if not self.verbose:
            if predicted_text not in self.banned_results:
                self.result_queue.put_nowait(predicted_text)
        else:
            if predicted_text not in self.banned_results:
                self.result_queue.put_nowait(result)

        if self.save_file:
            self.file.write(predicted_text)

    def listen_loop(self, dictate: bool = False, phrase_time_limit=None) -> None:

        """
        Override base method to yield result
        """

        self.recorder.listen_in_background(
            self.source, 
            self.__record_load, 
            phrase_time_limit=phrase_time_limit
        )
        self.logger.info("Listening...")
        threading.Thread(
            target=self.__transcribe_forever, daemon=True
        ).start()
        
        # while True:
        #     result = self.result_queue.get()
        #     if dictate:
        #         self.keyboard.type(result)
        #     else:
        #         #print(result)
        #         yield result
        
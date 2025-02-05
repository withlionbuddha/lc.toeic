import whisper
import os
import numpy as np

audio_path = r"F:/examination/workspace/lc.toeic/resource/mp3/AComprehensiveTOEICLCMP3/unit/03 PART 2/CH05_01.mp3"

class VOICEWIRTER :

    def __init__(self, file_path):
        self.file_path = file_path
    
    def writer(self) :
        
        self._check_file()
        
        audio_data = whisper.load_audio(self.file_path)
        
        self._check_autio_data(audio_data)
        print(f"{audio_data}")

        # Whisper 모델 로드 (기본 모델 사용)
        model = whisper.load_model("base")
        result = model.transcribe(audio_data, language="en")  # 영문 음성
            
        transcribed_text = result["text"]
        print(f"----------- text -------------")
        print(f"{transcribed_text}")

    def _check_file(self) :
        if isinstance(self.file_path, str):
            print(f"현재 작업 디렉토리: {os.getcwd()}")

            if os.path.exists(self.file_path):
                print(f": {self.file_path}")
                
                with open(self.file_path, "rb") as f :
                    print("파일이 정상적으로 열립니다.")
        else:
            print(f"FILE NOT FOUND : {self.file_path}")
    
    def _check_autio_data(self, audio) :
        if isinstance(audio, np.ndarray):
            print(f" shape: {audio.shape}")
        else:
            print(f"NOT SUPPORT: {type(audio)}")


if __name__ == "__main__":
    voiceWriter = VOICEWIRTER(audio_path)
    voiceWriter.writer()
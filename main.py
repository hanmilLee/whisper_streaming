import tkinter as tk
from tkinter import scrolledtext
import threading
import numpy as np
import sounddevice as sd
from whisper_online import asr_factory, OnlineASRProcessor
import soundfile as sf



# Whisper 설정
SAMPLING_RATE = 16000
args = lambda: None
args.backend = "faster-whisper"  # Backend 선택
args.lan = "ko"  # 언어 설정
args.model = "base"  # Whisper 모델 크기
args.model_cache_dir = "/Users/hanmillee/whisper_streaming/model_cache"
args.model_dir = "/Users/hanmillee/whisper_streaming/model"
args.vad = True  # Voice Activity Detection 사용
args.vac = True 
args.min_chunk_size = 1 
args.buffer_trimming = "segment"
args.buffer_trimming_sec = 3
args.task = 'transcribe'
asr, online = asr_factory(args)


# GUI 클래스 정의
class TranscriptApp:
    def __init__(self, root):
        self.root = root
        self.root.title("회의록 녹음 서비스")
        
        # 텍스트 영역
        self.text_area = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, width=80, height=20)
        self.text_area.grid(row=0, column=0, columnspan=2, padx=10, pady=10)
        
        # 버튼 정의
        self.start_button = tk.Button(self.root, text="녹음 시작", command=self.start_recording)
        self.start_button.grid(row=1, column=0, padx=10, pady=10)
        
        self.stop_button = tk.Button(self.root, text="녹음 중지", command=self.stop_recording, state=tk.DISABLED)
        self.stop_button.grid(row=1, column=1, padx=10, pady=10)
        
        self.play_button = tk.Button(self.root, text="선택 텍스트 재생", command=self.play_selected_audio)
        self.play_button.grid(row=2, column=0, columnspan=2, pady=10)
        
        # 상태 변수
        self.is_recording = False
        self.audio_buffer = np.array([], dtype=np.float32)
        self.transcripts = []
    
    def start_recording(self):
        self.is_recording = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.text_area.insert(tk.END, "녹음을 시작합니다...\n")
        threading.Thread(target=self.record_audio, daemon=True).start()
    
    def stop_recording(self):
        self.is_recording = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.text_area.insert(tk.END, "녹음을 중지합니다...\n")
    
    def record_audio(self):
        def callback(indata, frames, time, status):
            if status:
                print(f"상태 문제: {status}")
            if self.is_recording:
                self.audio_buffer = np.append(self.audio_buffer, indata.flatten())
                online.insert_audio_chunk(indata.flatten())
                result = online.process_iter()
                if result[2]:  # 텍스트 출력
                    self.transcripts.append(result)
                    self.text_area.insert(tk.END, f"[{result[0]:.2f}s - {result[1]:.2f}s] {result[2]}\n")
                    self.text_area.see(tk.END)
        
        # 사운드 스트림 설정
        with sd.InputStream(callback=callback, channels=1, samplerate=SAMPLING_RATE):
            while self.is_recording:
                sd.sleep(100)
    
    def play_selected_audio(self):
        try:
            selected_text = self.text_area.get(tk.SEL_FIRST, tk.SEL_LAST)
            for transcript in self.transcripts:
                if transcript[2].strip() in selected_text:
                    start_time, end_time = transcript[0], transcript[1]
                    audio_chunk = self.audio_buffer[int(start_time*SAMPLING_RATE):int(end_time*SAMPLING_RATE)]
                    sf.write("temp.wav", audio_chunk, SAMPLING_RATE)
                    data, _ = sf.read("temp.wav")
                    sd.play(data, samplerate=SAMPLING_RATE)
                    sd.wait()
                    break
        except tk.TclError:
            # Handle the error gracefully if no text is selected
            self.text_area.insert(tk.END, "텍스트를 선택하지 않았습니다. 재생하려면 텍스트를 선택하세요.\n")



# 애플리케이션 실행
if __name__ == "__main__":
    root = tk.Tk()
    app = TranscriptApp(root)
    root.mainloop()

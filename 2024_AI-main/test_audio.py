import os
import datetime
import sounddevice as sd
import wavio
from dotenv import load_dotenv

load_dotenv()

today = datetime.datetime.now().strftime("%Y%m%d%H%M")

recordfreq = 16000
duration = 5

fname = f"{today}_recording.wav"

print("Speak now...")
recording = sd.rec(int(duration * recordfreq), samplerate=recordfreq, channels=1)
sd.wait()
print(f"Recording complete! Type: {type(recording)}")

wavio.write(fname, recording, recordfreq, sampwidth=1)

rate, audio = wavio.read(fname)
sd.play(audio, samplerate=rate)
sd.wait()

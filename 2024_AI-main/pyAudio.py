import os 
import requests,uuid,json

import datetime
import sounddevice as sd

from scipy.io.wavfile import read
from scipy.io.wavfile import write
import wavio as wv

from dotenv import load_dotenv

load_dotenv()

Key = os.getenv("KEY1")

ENDPOINT= os.getenv("ENDPOINT")

URL = f"{ENDPOINT}"
today = datetime.datetime.now().strftime("sample")


# Record an audio file

freq = 16000
duration = 5

fname= f"{today}revprding.wav"

print("parla")

recording = sd.rec(int(duration*freq), samplerate = freq, channels = 1)

sd.wait()

print("OK!")

write(fname, freq,recording)

wv.write(fname, recording, freq, sampwidth=1)

headers = {
    'Ocp-Apim-Subscription-Key': Key,
    'Content-Type':f'codecs=audio/pcm; samplearate={freq}',
    'Accept':'text/json'

}

params = {
    'language':'it-it'
}

buf = open(fname,'rb')

res = requests.post(URL,data=buf, headers=headers, params=params)
print(f"Res: {res.status_code}.{res.reason}")
res.raise_for_status()

print(res.json())
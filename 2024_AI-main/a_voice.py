import os
import requests
from scipy.io.wavfile import read
import sounddevice as sd
from dotenv import load_dotenv

load_dotenv()

# Assuming you have keys and location stored in your .env file
key = os.getenv("KEY1")
location = os.getenv("LOCATION")

endpoint = f"https://{location}.tts.speech.microsoft.com/cognitiveservices/v1"
path = ""

brhead = {
    'Content-type': 'application/x-www-form-urlencoded',
    'Content-Length': '0',
    'Ocp-Apim-Subscription-Key': key
}

token = ''
rtoken = requests.post(f"https://{location}.api.cognitive.microsoft.com/sts/v1.0/issueToken", headers=brhead)

if rtoken.status_code == 200:
    token = rtoken.text

constructed_url = endpoint + path

auth = f"Bearer {token}"
headers = {
    'Ocp-Apim-Subscription-Key': key,
    'X-Microsoft-OutputFormat': 'riff-8khz-8bit-mono-mulaw',
    'Content-type': 'application/ssml+xml',
    'Authorization': auth,
    'User-Agent': 'apicaller'
}

body = '''<speak version='1.0' xml:lang='it-IT'>
<voice xml:lang='it-IT' xml:gender='Female' name='it-IT-FabiolaNeural'>
Buongiorno! Oggi è un bel giorno per fare qualcosa
</voice>
<voice xml:lang='it-IT' xml:gender='Female' name='it-IT-ElsaNeural'>
Prego dire un comando. Ad esempio "Invia una email a Marco".
</voice>
</speak>'''

request = requests.post(constructed_url, data=body.encode('utf-8'), headers=headers)

if request.status_code == 200:
    with open('sample.wav', 'wb') as audio_file:
        audio_file.write(request.content)
        print("\nStatus code:" + str(request.status_code) + "\nYour TTS is ready for playback. \n")
else:
    print("\nStatus code:" + str(request.status_code) + "\nSomething went wrong. Chekc your Subscription")
    print("Reason " + str(request.reason) + "\n")

#print(json.dumps(response.json(), sort_keys=True))
rate, audio = read("sample.wav")
# Play the audio
sd.play(audio, samplerate=rate)
sd.wait()

print("Done")

from langdetect import detect
from translate import Translator
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials

# Function to identify the language
def identify_language(text):
    return detect(text)

# Function to get the description of an image
def get_image_description(image_url):

    # Create a Computer Vision client
    computer_vision_client = ComputerVisionClient(endpoint="https://labai900bl.cognitiveservices.azure.com/", 
                                                credentials=CognitiveServicesCredentials('ec1e078978384920bee0685f891b10b4'))

    # Get the description of an image
    image_description = computer_vision_client.describe_image(image_url).captions[0].text

    return image_description

# Function to translate text
def translate_text(text, target_language):
    translator = Translator(to_lang=target_language)
    translation = translator.translate(text)
    return translation

# Main function
if __name__ == '__main__':
    # Interagisci con l'utente
    user_input = input("Enter text: ")

    # Identifica la lingua nativa dell'utente
    native_language = identify_language(user_input)

    # Istuzioni di base nelle principali lingue europee
    if native_language == 'it':
        print("Descrizione dell'immagine.")
        print("Per favore, fornisci l'URL dell'immagine.")
    elif native_language == 'fr':
        print("Description de l'image.")
        print("Veuillez fournir l'URL de l'image.")
    elif native_language == 'es':
        print("Descripción de la imagen.")
        print("Por favor, proporcione la URL de la imagen.")
    elif native_language == 'de':
        print("Beschreibung des Bildes.")
        print("Bitte geben Sie die URL des Bildes an.")
    elif native_language == 'ru':
        print("Описание изображения.")
        print("Пожалуйста, укажите URL изображения.")
    elif native_language == 'ja':
        print("画像の説明。")
        print("画像のURLを指定してください。")

    # Ottieni la descrizione dell'immagine scelta dall'utente
    image_url = input("Image URL: ")
    image_description = get_image_description(image_url)

    print("Original description:", image_description)

    # Traduci la descrizione dell'immagine nelle lingue richieste
    languages_to_translate = ['it', 'fr', 'es', 'de', 'ru', 'ja']
    for lang in languages_to_translate:
        if lang != native_language:
            translation = translate_text(image_description, lang)
            print(f"Translated description in {lang}: {translation}")

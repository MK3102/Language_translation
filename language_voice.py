from transformers import MarianMTModel, MarianTokenizer
import requests
from bs4 import BeautifulSoup
import pyttsx3
import speech_recognition as sr

LANGUAGES = {
    "english": "en",
    "french": "fr",
    "spanish": "es",
    "chinese": "zh",
    "hindi": "hi",
    "russian": "ru",
    "sanskrit": "sa",
    "urdu": "ur",
    "arabic": "ar"
}

def load_model(src_lang, tgt_lang):
    model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
    try:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        return model, tokenizer
    except Exception:
        return None, None

def translate_text(text, src_lang, tgt_lang):
    if not text.strip() or src_lang not in LANGUAGES.values() or tgt_lang not in LANGUAGES.values():
        return None
    
    model, tokenizer = load_model(src_lang, tgt_lang)
    if model is None or tokenizer is None:
        return google_translate(text, src_lang, tgt_lang)

    tokenized_text = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated_tokens = model.generate(**tokenized_text)
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

def google_translate(text, src_lang, tgt_lang):
    try:
        url = f"https://translate.google.com/m?sl={src_lang}&tl={tgt_lang}&q={text.replace(' ', '%20')}"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        translated_text = soup.find("div", class_="result-container").text
        return translated_text
    except Exception:
        return None

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google(audio)
    except:
        return None

if __name__ == "__main__":
    print("Supported languages:", ", ".join(LANGUAGES.keys()))

    print("Say the text to translate:")
    text = listen()
    
    if not text:
        print("Could not recognize speech.")
    else:
        print(f"You said: {text}")

        print("Say the source language (e.g., 'english', 'french'):")
        src_lang_name = listen()
        
        print("Say the target language (e.g., 'spanish', 'chinese'):")
        tgt_lang_name = listen()

        if src_lang_name and tgt_lang_name:
            src_lang_name = src_lang_name.lower().strip()
            tgt_lang_name = tgt_lang_name.lower().strip()

            if src_lang_name not in LANGUAGES or tgt_lang_name not in LANGUAGES:
                print("Error: Unsupported language.")
            else:
                translated_text = translate_text(text, LANGUAGES[src_lang_name], LANGUAGES[tgt_lang_name])
                if translated_text:
                    print("Translated text:", translated_text)
                    speak(translated_text)
                else:
                    print("Translation failed.")

from transformers import MarianMTModel, MarianTokenizer

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
        return None

    tokenized_text = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated_tokens = model.generate(**tokenized_text)
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

def translate_with_intermediate(text, src_lang, tgt_lang):
    if f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}' in supported_models():
        return translate_text(text, src_lang, tgt_lang)

    if src_lang != "en" and tgt_lang != "en":
        text = translate_text(text, src_lang, "en")
        if text:
            return translate_text(text, "en", tgt_lang)
    
    return None

def supported_models():
    return {
        "Helsinki-NLP/opus-mt-en-fr",
        "Helsinki-NLP/opus-mt-en-es",
        "Helsinki-NLP/opus-mt-en-zh",
        "Helsinki-NLP/opus-mt-en-hi",
        "Helsinki-NLP/opus-mt-en-ru",
        "Helsinki-NLP/opus-mt-en-sa",
        "Helsinki-NLP/opus-mt-en-ur",
        "Helsinki-NLP/opus-mt-en-ar",
        "Helsinki-NLP/opus-mt-fr-en",
        "Helsinki-NLP/opus-mt-es-en",
        "Helsinki-NLP/opus-mt-zh-en",
        "Helsinki-NLP/opus-mt-hi-en",
        "Helsinki-NLP/opus-mt-ru-en",
        "Helsinki-NLP/opus-mt-sa-en",
        "Helsinki-NLP/opus-mt-ur-en",
        "Helsinki-NLP/opus-mt-ar-en",
    }

if __name__ == "__main__":
    print("Supported languages:", ", ".join(LANGUAGES.keys()))

    text = input("Enter the text to translate: ").strip()
    src_lang_name = input("Enter the source language (e.g., 'english', 'french'): ").strip().lower()
    tgt_lang_name = input("Enter the target language (e.g., 'spanish', 'chinese'): ").strip().lower()

    if src_lang_name not in LANGUAGES or tgt_lang_name not in LANGUAGES:
        print("Error: Unsupported language.")
    else:
        translated_text = translate_with_intermediate(text, LANGUAGES[src_lang_name], LANGUAGES[tgt_lang_name])
        if translated_text:
            print("Translated text:", translated_text)

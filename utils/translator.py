from transformers import pipeline
from langdetect import detect

# Preload commonly used translators
SUPPORTED_MODELS = {
    "ar": "Helsinki-NLP/opus-mt-ar-en",
}

translators = {
    lang: pipeline("translation", model=model_name)
    for lang, model_name in SUPPORTED_MODELS.items()
}

# if the text is already in English return it, if not translate to English
def translate_to_english(text: str) -> str:
    try:
        lang = detect(text)
        if lang == "en":
            return text
        if lang in translators:
            return translators[lang](text)[0]['translation_text']
        return text  # Fallback for unsupported
    except Exception:
        return text
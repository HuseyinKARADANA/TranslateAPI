from flask import Flask, request, jsonify
from langdetect import detect
from transformers import MarianMTModel, MarianTokenizer

app = Flask(__name__)

# Çeviri modelini cache’liyoruz
models_cache = {}

def get_model(src_lang, tgt_lang):
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    if model_name not in models_cache:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        models_cache[model_name] = (tokenizer, model)
    return models_cache[model_name]

@app.route('/translate', methods=['POST'])
def translate_text():
    data = request.get_json()
    text = data.get('text', '')
    target_lang = data.get('target_lang', '')

    if not text or not target_lang:
        return jsonify({'error': 'text ve target_lang zorunludur'}), 400

    try:
        source_lang = detect(text)
        tokenizer, model = get_model(source_lang, target_lang)

        tokens = tokenizer.prepare_seq2seq_batch([text], return_tensors="pt")
        translated = model.generate(**tokens)
        translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)

        return jsonify({
            'source_lang': source_lang,
            'translated_text': translated_text
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8003)


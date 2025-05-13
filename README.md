# TranslateAPI
gerekli kütüphaneler
pip install flask langdetect transformers torch sentencepiece

endpoint: /translate
gönderilen JSON
{
  "text": "merhaba nasılsın",
  "target_lang": "fr"
}

Gelen JSON da mevcut gönderilen dil bilgisi ve çevrilmiş metin dönüyor



# Jarvis - Kişisel Sesli Asistan

Bu proje, Iron Man filmindeki Jarvis'ten esinlenerek geliştirilmiş kişisel bir sesli asistan uygulamasıdır. MacOS üzerinde çalışacak şekilde tasarlanmıştır.

## Özellikler

- Sürekli ses dinleme ve konuşmacı tanıma
- Özelleştirilmiş ses ile yanıt verme
- Sistem kontrolü (parlaklık, ses, uygulamalar vb.)
- Konuşma geçmişi
- Konuşmacı doğrulama

## Gereksinimler

- Python 3.8+
- PyAudio
- OpenAI Whisper
- TTS (Text-to-Speech)
- Diğer gereksinimler requirements.txt dosyasında listelenmiştir

## Kurulum

1. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

2. Ses profilinizi oluşturun:
- Jarvis'i ilk kez çalıştırdığınızda, sesinizi tanıması için bir profil oluşturmanız gerekecektir.
- Birkaç saniye boyunca konuşmanız istenecektir.

## Kullanım

Asistanı başlatmak için:
```bash
python jarvis.py
```

## Güvenlik

- Sistem komutları için sudo yetkisi gerekebilir
- Ses profili verileriniz lokalde saklanır
- Tüm işlemler yerel olarak gerçekleştirilir

## Lisans

MIT

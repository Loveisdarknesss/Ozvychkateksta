from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import soundfile as sf
from gtts import gTTS
import os

text = 'Хочу всю жизнь тебя любить,О грусти навсегда забыть, Ведь счастье — только лишь с тобой,Любви не нужно мне иной, Хочу обнять тебя скорей, Ведь в мире нет тебя нежней, И больше нет прекрасных слов, Чтоб выразить мою любовь.'
tts = gTTS(text, lang='ru')
tts.save('output.wav')
os.system('output.wav')



def text_to_audio(wav2vec2_model='facebook/wav2vec2-base-960h', voice_preset='v2/ru_speaker_3'):
    processor = Wav2Vec2Processor.from_pretrained(wav2vec2_model)
    model = Wav2Vec2ForCTC.from_pretrained(wav2vec2_model)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    text = 'Хочу всю жизнь тебя любить,О грусти навсегда забыть, Ведь счастье — только лишь с тобой,Любви не нужно мне иной, Хочу обнять тебя скорей, Ведь в мире нет тебя нежней, И больше нет прекрасных слов, Чтоб выразить мою любовь.'

    inputs = processor(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    predicted_text = processor.batch_decode(predicted_ids)

    # Сохранение в аудиофайл
    output_file = f'{voice_preset.split("/")[1]}.wav'
    sf.write(output_file, predicted_text[0], 16000)  # 16000 - частота дискретизации


def main():
    text_to_audio()


if __name__ == '__main__':
    main()

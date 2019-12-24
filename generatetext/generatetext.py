import keras
from keras import layers
import numpy as np
import random

text = ''
with open('LordOfTheRings1.txt', 'r', encoding='utf-8') as book:
    text = book.read().lower().replace('\n', ' ')

print('Длинна текста:', len(text))

maxlen = 100      # Длинна извлекаемых последовательностей
step = 3         # Шаг, с которым будут выбираться последовательности
sentences = []   # Список для хранения извлечённых последовательностей
next_chars = []  # Цели (символы, идущие после последовательностей)

for index in range(0, len(text) - maxlen, step):
    sentences.append(text[index: index + maxlen])
    next_chars.append(text[index + maxlen])

print('Количество полученных предложений:', len(sentences))

chars = sorted(list(set(text)))                                   # Создание списка со всеми символами
print('Всего различных символов найдено:', len(chars))
char_indices = dict((char, chars.index(char)) for char in chars)  # Создание словаря с индексами символов в списке chars

print('Векторизация текста....')
vectorized_sentences = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
vectorized_next_chars = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for sent_index, sentence in enumerate(sentences):
    for chr_index, char in enumerate(sentence):
        vectorized_sentences[sent_index, chr_index, char_indices[char]] = 1
    vectorized_next_chars[sent_index, char_indices[next_chars[sent_index]]] = 1

model = keras.models.Sequential()
model.add(layers.LSTM(128, input_shape=(maxlen, len(chars))))
model.add(layers.Dense(len(chars), activation='softmax'))

optimizer = keras.optimizers.RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


def get_next_char(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate_text(start_text, new_chars, temperature=1.0):
    global maxlen, chars, char_indices, model
    out = start_text
    start_text = start_text[len(start_text) - maxlen:]
    for c in range(new_chars):
        sampled = np.zeros((1, maxlen, len(chars)))
        for chr_index, char in enumerate(start_text):
            sampled[0, chr_index, char_indices[char]] = 1

        preds = model.predict(sampled, verbose=0)[0]
        next_index = get_next_char(preds, temperature)
        next_char = chars[next_index]

        start_text += next_char
        start_text = start_text[1:]

        out += next_char
    return out


start_text = '''Арагорн бежал вверх по склону, часто наклоняясь и внимательно осматривая землю.'''
start_text = start_text.lower().replace('\n', ' ')
for epoch in range(1, 60):
    print('Эпоха номер', epoch)
    model.fit(vectorized_sentences, vectorized_next_chars, batch_size=128, epochs=1)

    for temperature in [0.2, 0.5, 1.0, 1.2]:
        print('-------- temperature:', temperature)
        print(generate_text(start_text, 100, temperature))

model.save('generatetext_model.h5')

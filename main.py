import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from gtts import gTTS
from dataset import dataset


texts = list(dataset.keys())
labels = list(dataset.values())

vectorizer = CountVectorizer()
vectors = vectorizer.fit_transform(texts)

clf = LogisticRegression()
clf.fit(vectors, labels)

def text_to_speech(text):
    tts = gTTS(text=text, lang='ru')
    tts.save('output.mp3')
    os.system('output.mp3')

text = input('Введите ваш вопрос: ')
text_vector = vectorizer.transform([text]).toarray()
answer = clf.predict(text_vector)[0]
print(answer)
text_to_speech(answer)
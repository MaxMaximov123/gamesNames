import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


def convert_text_to_vector(text):
	text = text.replace(' ', '').lower()
	exp_d = [0] * 64
	for i in range(len(text)):
		exp_d[i] = 1 / ord(text[i])
	return exp_d


def siamese_network(input_dim):
	# Входные тензоры для двух строк
	input_a = tf.keras.Input(shape=(input_dim,))
	input_b = tf.keras.Input(shape=(input_dim,))

	# Общая ветвь нейронной сети
	shared_network = tf.keras.Sequential([
		layers.Dense(128, activation='relu'),
		layers.Dense(128, activation='relu')
	])

	# Получение векторных представлений для каждой строки
	encoded_a = shared_network(input_a)
	encoded_b = shared_network(input_b)

	# Вычисление расстояния между векторами
	distance = tf.keras.layers.Lambda(
		lambda x: tf.math.abs(x[0] - x[1])
	)([encoded_a, encoded_b])

	# Выходной слой
	output = layers.Dense(1, activation='sigmoid')(distance)

	# Создание модели сиамской нейронной сети
	model = tf.keras.Model(inputs=[input_a, input_b], outputs=output)

	return model


# Создание экземпляра модели
model = siamese_network(input_dim=64)

# Компиляция модели
model.compile(optimizer='adam', loss='binary_crossentropy')


def forecast(n1, n2):
	return model.predict([np.array([convert_text_to_vector(n1)]), np.array([convert_text_to_vector(n2)])])[0]


# Генерация случайных данных для обучения
# input_a = np.random.random((1000, 100))
# input_b = np.random.random((1000, 100))
# labels = np.random.randint(2, size=(1000,))

# print(labels)

# Обучение модели
a = 1
if a:
	import convert_funtions

	input_a = convert_funtions.input_a
	input_b = convert_funtions.input_b

	labels = convert_funtions.labels
	model.fit([input_a, input_b], labels, epochs=500, batch_size=32)
	model.save('../models/model7.h5')
else:
	model.load_weights('../models/model7.h5')

for n1, n2 in [
	['Формис II (жен)', 'Ledec nad Sazavou'],
	['Формис II (жен)', 'Formis-2 (w)'],
	['Real Sociedad (Nicolas_Rage)', 'Реал Сосьедад (Nicolas_Rage)'],
	['Parentini Vallega Montebruno G/Ruggeri J', 'Parentini Vallega Montebruno / Ruggeri'],
	['Formis-2 (w)', 'Формис II (жен)'],
	['Formis-2 (w)', 'Formis-2 (w)'],
	['спрпропро', 'нггвгнынгоен'],
	['Real Sociedad (Nicolas_Rage)', 'Real Sociedad (Nicolas_Rage) Esports'],
	['Austin Peay (w)', 'Austin Peay Women'],
	['Ferroviaria San Paolo', 'Ферровиария СП'],
	['Ferroviaria San Paolo', 'Ferroviaria SP'],
	['Elizabeth Ionescu', 'Elizabeth Ionescu (USA)']
]:
	print(n1, '+', n2, ' -> ', forecast(n1, n2))


n1, n2 = input('Введите имена: ').split('!')
while n1 != '0' and n2 != '0':
	print(forecast(n1, n2))
	n1, n2 = input('Введите имена: ').split('!')
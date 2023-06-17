import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


def convert_text_to_vector(text):
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

model.load_weights('../models/model3.h5')


def forecast(n1, n2):
    return model.predict([np.array([convert_text_to_vector(n1)]), np.array([convert_text_to_vector(n2)])])


print(forecast('Formis', 'Formis'))
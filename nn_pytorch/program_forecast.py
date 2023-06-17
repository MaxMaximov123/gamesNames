import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton
from PyQt5.QtWidgets import QLabel, QLineEdit, QComboBox, QProgressBar
from PyQt5.QtCore import Qt

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


class Example(QWidget):
	def __init__(self):
		self.f = True
		self.models_files = [i for i in os.listdir('../models') if i[-2:] == 'h5']
		self.model = self.siamese_network(input_dim=64)
		self.model.compile(optimizer='adam', loss='binary_crossentropy')
		self.model.load_weights(f'../models/{self.models_files[-1]}')
		super().__init__()
		self.initUI()


	def initUI(self):
		self.setGeometry(500, 500, 450, 60)
		self.setWindowTitle('Сравнение названий')

		self.btn = QPushButton('->', self)
		self.btn.clicked.connect(self.start)
		self.btn.move(300, 0)
		self.btn.resize(50, 30)

		self.n1 = QLineEdit(self)
		self.n1.move(0, 0)
		self.n1.resize(150, 30)
		self.n1.setText('Austin Peay (w)')

		self.n2 = QLineEdit(self)
		self.n2.move(150, 0)
		self.n2.resize(150, 30)
		self.n2.setText('Austin Peay Women')

		self.res = QLineEdit(self)
		self.res.move(350, 0)
		self.res.setEnabled(False)
		self.res.resize(100, 30)
		self.res.setText('0')

		self.combobox = QComboBox(self)
		self.combobox.addItems(self.models_files)
		self.combobox.move(0, 30)
		self.combobox.resize(100, 30)
		self.combobox.setCurrentIndex(len(self.models_files)-1)
		self.combobox.currentTextChanged.connect(self.update_model)

		self.pb = QProgressBar(self)
		self.pb.move(100, 30)
		self.pb.resize(350, 30)
		self.pb.setValue(0)

	def update_model(self):
		self.model.load_weights(f'../models/{self.combobox.currentText()}')

	def forecast(self, n1, n2):
		return self.model.predict(
			[
				np.array([self.convert_text_to_vector(n1)]),
			 	np.array([self.convert_text_to_vector(n2)])
			])[0][0]

	def convert_text_to_vector(self, text):
		text = text.replace(' ', '').lower()
		exp_d = [0] * 64
		for i in range(len(text)):
			exp_d[i] = 1 / ord(text[i])
		return exp_d

	def siamese_network(self, input_dim):
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

	def start(self):
		n1 = self.n1.text()
		n2 = self.n2.text()
		self.res.setText(str(self.forecast(n1, n2)))

	def keyPressEvent(self, event):
		if event.key() == Qt.Key_Return:
			print(999)
			self.start()


if __name__ == '__main__':
	app = QApplication(sys.argv)
	ex = Example()
	ex.show()
	sys.exit(app.exec())

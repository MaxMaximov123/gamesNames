from pprint import pprint
import torch
import sqlite3
from tqdm import tqdm
import numpy as np
import numpy as np

# Проверяем доступность GPU


sqlite_connection = sqlite3.connect('../data/names.db')
cursor = sqlite_connection.cursor()

sqlite_select_query = """SELECT * from data"""
cursor.execute(sqlite_select_query)
train_data = []
target_labels = []
input_a = []
input_b = []
labels = []


def convert_text_to_vector(text):
    text = text.replace(' ', '').lower()
    exp_d = [0] * 64
    for i in range(len(text)):
        exp_d[i] = 1 / ord(text[i])
    return exp_d


def convert_game_name(name1, name2, st, b1, b2):
    exp_d = [0] * 128
    exp_d[0] = b1 / 10
    exp_d[127] = b2 / 10
    for i in range(len(name1)):
        exp_d[i + 1] = 1 / ord(name1[i])
    for i in range(len(name2)):
        exp_d[126 - i] = 1 / ord(name2[i])
    return exp_d


def create_dataset():
    global input_b, input_a, labels
    for i in tqdm(cursor.fetchall()):
        # train_data.append(convert_game_name(i[0], i[1], 0, i[3], i[4]))
        # target_labels.append([i[2]])

        input_a.append(convert_text_to_vector(i[0]))
        input_b.append(convert_text_to_vector(i[1]))

        labels.append(i[2])



create_dataset()
input_a = np.array(input_a)
input_b = np.array(input_b)

labels = np.array(labels)
# train_data = np.array(train_data)
# target_labels = np.array(target_labels)


if __name__ == '__main__':
    pprint(input_a)
    pprint(input_b)
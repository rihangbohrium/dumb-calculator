import numpy as np
import random

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Dense, Dropout, SimpleRNN, RepeatVector
from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback
tf.config.run_functions_eagerly(True)

ops = ['+', '-'] 
all_chars = '0123456789' + ''.join(ops)
all_chars

num_features = len(all_chars)
char_to_index = dict((c, i) for i, c in enumerate(all_chars)) # tokenize
index_to_char = dict((i, c) for i, c in enumerate(all_chars))

LO, HI = 0, 1000
def generate_data(lo=LO, hi=HI):
    n1 = np.random.randint(lo, hi+1)
    n2 = np.random.randint(lo, hi+1)
    op = random.choice(ops)
    if (op == '/' and n2 == 0):
        n2 = 1 # jankly avoid div by 0 err
    example = str(n1) + op + str(n2)
    label = 0
    if op == '+':
        label = n1 + n2
    elif op == '-':
        label = n1 - n2
    elif op == '*':
        label = n1 * n2
    elif op == '/':
        label = n1 // n2
    return example, str(label)

# Recurrent NN for variable vectors, both input and output
hidden_units = 128
max_time_steps = 2 * 3 + 1 # max length of input

model = Sequential([
    SimpleRNN(hidden_units, input_shape=(None, num_features)),
    RepeatVector(max_time_steps), # get singular vec representation
    # decoder starts here:
    SimpleRNN(hidden_units, return_sequences=True),
    TimeDistributed(Dense(num_features, activation='softmax'))
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

def vectorize_example(example, label):
    x = np.zeros((max_time_steps, num_features))
    y = np.zeros((max_time_steps, num_features))

    diff_x = max_time_steps - len(example)
    diff_y = max_time_steps - len(label) 
        
    for i, c in enumerate(example):
        x[i + diff_x, char_to_index[c]] = 1
    for i in range(diff_x):
        x[i, char_to_index['0']] = 1
    for i, c in enumerate(label):
        y[i + diff_y, char_to_index[c]] = 1
    for i in range(diff_y):
        y[i, char_to_index['0']] = 1
    
    return x, y

def devectorize_example(example):
    result = [index_to_char[np.argmax(vec)] for i, vec in enumerate(example)]
    return ''.join(result)

def create_dataset(num_examples=2000):

    x_train = np.zeros((num_examples, max_time_steps, num_features))
    y_train = np.zeros((num_examples, max_time_steps, num_features))

    for i in range(num_examples):
        e, l = generate_data()
        x, y = vectorize_example(e, l)
        x_train[i] = x
        y_train[i] = y
    
    return x_train, y_train
def del_leading_zeros(s):
    return s.lstrip('0')
    
def calc_example(new_model, example:str):
    temp = np.zeros((1, max_time_steps, num_features))
    example, label = vectorize_example(example, '')
    temp[0] = example
    
    pred = new_model.predict(temp)
    r = devectorize_example(pred[0])
    return del_leading_zeros(r)


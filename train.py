# -*- coding: utf-8 -*-
import os
import re
import shutil
import string
import tensorflow as tf
import csv
import numpy as np
import random
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers
from tensorflow.keras import losses
from dask import dataframe as dd

sample_df = dd.read_csv(
    f"output*.csv",
    delimiter=',',
    header = 0,
    names=["np", "genre"],
    dtype={'loc': 'object'},
    engine='python')
sample_df.np = sample_df.np.str.replace('[^a-zA-Z ]', '')

df = sample_df.compute(num_workers=10)
df = df.sample(frac=1)
df = df.drop_duplicates(subset=['np'], keep='last')

vocab_size = 1000
embedding_dim = 64
max_length = 20
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_portion = .8

features = df['np'].to_list()
labels = df['genre'].to_list()

train_size = int(len(features) * training_portion) 

train_sentences = features[:train_size]
train_labels = labels[:train_size]

validation_sentences = features[train_size:] 
validation_labels = labels[train_size:]


tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_padded = pad_sequences(train_sequences, padding=padding_type, maxlen=max_length)

validation_sequences = tokenizer.texts_to_sequences(validation_sentences)
validation_padded = pad_sequences(validation_sequences, padding=padding_type, maxlen=max_length)

label_tokenizer = Tokenizer() 
label_tokenizer.fit_on_texts(labels)

training_label_seq = np.array(label_tokenizer.texts_to_sequences(train_labels))
validation_label_seq = np.array(label_tokenizer.texts_to_sequences(validation_labels))

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)), 
    tf.keras.layers.Dense(64,kernel_regularizer=tf.keras.regularizers.l2(0.001), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(6, activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

num_epochs = 15
history = model.fit(train_padded, training_label_seq, epochs=num_epochs, validation_data=(validation_padded, validation_label_seq), verbose=2)

fname = "model.h5"
model.save(fname)
print('All done')
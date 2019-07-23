import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf
import translator.data
import numpy as np


def create_model(settings, vocabulary_size):
    inputs = tf.keras.layers.Input((settings['encoder_inputs'], ), name="encoder_inputs")
    embedding = tf.keras.layers.Embedding(vocabulary_size + 1, settings['embedding_size'])(inputs)
    encoder = tf.keras.layers.GRU(settings['units'], name="encoder", return_sequences=True)(embedding)
    decoder = tf.keras.layers.GRU(settings['units'], name='decoder', return_sequences=True)(encoder)
    decoder_attention = tf.keras.layers.Attention(name="decoder_attention")([encoder, decoder])
    outputs = tf.keras.layers.Dense(vocabulary_size, activation='softmax', name="outputs")(decoder_attention)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(settings['optimizer'], settings['loss'], settings['metrics'])
    return model


if __name__ == '__main__':
    config = {
            "encoder_inputs": 64,
            "embedding_size": 256,
            "units": 128,
            "encoder_cells": 3,
            "decoder_cells": 3,
            "optimizer": "rmsprop",
            "loss": "sparse_categorical_crossentropy",
            "batch_size": 32,
            "epochs": 3,
            "metrics": [
            ]
        }
    
    #tf.keras.utils.plot_model(m, "model.png", show_shapes=True)
    
    raw_x = translator.data.load_corpus("../datasets/raw/train_ger_x.txt")
    raw_y = translator.data.load_corpus("../datasets/raw/train_ger_y.txt")
    german_vocabulary = translator.data.make_vocabulary(raw_x, False, 50000)
    english_vocabulary = translator.data.make_vocabulary(raw_y, False, 50000)
    
    train_generator = translator.data.Dataset(raw_x, raw_y, german_vocabulary, english_vocabulary, config)
    test_generator = translator.data.Dataset(translator.data.load_corpus("../datasets/raw/test_ger_x.txt"),
                                             translator.data.load_corpus("../datasets/raw/test_ger_x.txt"),
                                             german_vocabulary, english_vocabulary, config)
    m = create_model(config, 50000)
    with tf.Session() as session:
        tf.keras.backend.set_session(session)
        m.fit_generator(train_generator, validation_data=test_generator, epochs=10, callbacks=[
            tf.keras.callbacks.ModelCheckpoint("models/weights.{epoch:02d}-{val_loss:.3f}.hdf5", 'val_loss'),
            tf.keras.callbacks.EarlyStopping('val_loss', 0.001, 2),
            tf.keras.callbacks.TerminateOnNaN(), tf.keras.callbacks.ReduceLROnPlateau(patience=2)])

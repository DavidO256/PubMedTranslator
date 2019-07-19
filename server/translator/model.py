import tensorflow as tf
import translator.data
import numpy as np


def create_model(settings, vocabulary_size):
    inputs = tf.keras.layers.Input(settings['encoder_inputs'], name="encoder_inputs")
    embedding = tf.keras.layers.Embedding(vocabulary_size, settings['embedding_size'])(inputs)
    encoder = tf.keras.layers.RNN([tf.keras.layers.LSTMCell(settings['units'], name="encoder_cell_%d" % n)
                                   for n in range(settings['encoder_cells'])], name="encoder",
                                   return_sequences=True)(embedding)

    score = tf.keras.layers.Dense(settings['embedding_size'], tf.keras.activations.tanh)

    decoder = tf.keras.layers.RNN([tf.keras.layers.LSTMCell(settings['units'], name="decoder_cell_%d" % n)
                                   for n in range(settings['decoder_cells'])], name="decoder",
                                   return_sequences=True)(encoder)
    decoder_attention = tf.keras.layers.Attention(name="decoder_attention")([encoder, decoder])
    decoder_outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(vocabulary_size, name="outputs"))(decoder_attention)
    outputs = tf.keras.layers.Softmax()(decoder_outputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(settings['optimizer'], settings['loss'], settings['metrics'])
    return model


if __name__ == '__main__':
    config = {
            "encoder_inputs": 64,
            "embedding_size": 512,
            "units": 128,
            "encoder_cells": 3,
            "decoder_cells": 3,
            "optimizer": "adam",
            "loss": "mse",
            "metrics": [
                "accuracy"
            ]
        }
    m = create_model(config, 50)
    tf.keras.utils.plot_model(m, "model.png", show_shapes=True)

    raw_x = translator.data.load_corpus("../datasets/temp/raw/train_ger_x.txt")
    raw_y = translator.data.load_corpus("../datasets/temp/raw/train_ger_y.txt")
    german_vocabulary = translator.data.make_vocabulary(raw_x, False, 50)
    english_vocabulary = translator.data.make_vocabulary(raw_y, False, 50)
    print("German Vocab:", len(german_vocabulary), "\nEnglish Vocab:", len(english_vocabulary))
    x = np.asarray([translator.data.process_inputs(sample, german_vocabulary, config) for sample in raw_x])
    y = np.asarray([translator.data.process_outputs(sample, english_vocabulary, config) for sample in raw_y])
    print(x.shape, "\n", y.shape)
    m.fit(x, y)
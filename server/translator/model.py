import tensorflow as tf
import translator.data
import numpy as np
    

def create_default_model(settings, inputs_vocabulary_size, outputs_vocabulary_size):
    inputs = tf.keras.layers.Input((settings['encoder_inputs'], ), name="encoder_inputs")
    embedding = tf.keras.layers.Embedding(inputs_vocabulary_size + 1, settings['embedding_size'])(inputs)
    encoder = tf.keras.layers.GRU(settings['units'], name="encoder", return_sequences=True)(embedding)
    decoder = tf.keras.layers.GRU(settings['units'], name='decoder', return_sequences=True)(encoder)
    decoder_attention = tf.keras.layers.Attention(name="decoder_attention")([encoder, decoder])
    outputs = tf.keras.layers.Dense(outputs_vocabulary_size, activation='softmax', name="vocabulary")(decoder_attention)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(settings['optimizer'], settings['loss'], settings['metrics'])
    return model


class TranslationModel:

    def __init__(self, settings):
        self.x_vocabulary = None
        self.y_vocabulary = None
        self.model = None
        self.settings = settings

    def fit(self, x_corpus, y_corpus, epochs=1, validation_x=None, validation_y=None):
        self.x_vocabulary, self.y_vocabulary = (translator.data.make_vocabulary(corpus, self.settings['case_sensitive'], self.settings['vocabulary_cutoff']) for corpus in [x_corpus, y_corpus])
        if self.x_vocabulary is None or self.y_vocabulary is None:
            raise ValueError("Error fitting model: x and y vocabulary must be initialized before or during fitting.")
        train = translator.data.Dataset(x_corpus, y_corpus, self.x_vocabulary, self.y_vocabulary, self.settings)
        validation = translator.data.Dataset(validation_x, validation_y, self.x_vocabulary, self.y_vocabulary, self.settings)\
                        if validation_x is not None and validation_y is not None else None
        if self.model is None:
            self.model = create_default_model(self.settings, len(self.x_vocabulary), len(self.y_vocabulary))
        tf.keras.utils.plot_model(self.model, "model.png", show_shapes=True)
        with tf.Session() as session:
            tf.keras.backend.set_session(session)
            self.model.fit_generator(train, validation_data=validation, epochs=epochs, callbacks=[
                tf.keras.callbacks.ModelCheckpoint("models/weights.{epoch:02d}-{val_loss:.3f}.hdf5", 'val_loss'),
                tf.keras.callbacks.EarlyStopping('val_loss', 0.001, 2),
                tf.keras.callbacks.TerminateOnNaN(), tf.keras.callbacks.ReduceLROnPlateau(patience=2)])

    def predict(x, **kwargs):
        if isinstance(x, str):
            return self.predict([x], **kwargs)
        return [self.model.predict(translator.data.process_x(sample, self.x_vocabulary, self.settings))
                for sample in x]

    def predict_raw(self, *args, **kwargs):
        return [translator.data.convert_raw(y, self.y_vocabulary, self.settings)
                for y in self.predict(*args, **kwargs)]
    
if __name__ == '__main__':
    config = {
            "encoder_inputs": 64,
            "embedding_size": 256,
            "units": 128,
            "encoder_cells": 3,
            "decoder_cells": 3,
            "optimizer": "rmsprop",
            "loss": "categorical_crossentropy",
            "batch_size": 32,
            "epochs": 3,
            "case_sensitive": False,
            "vocabulary_cutoff": 50000,
            "metrics": [
            ]
        }
    
    
    raw_x = translator.data.load_corpus("../datasets/raw/train_ger_x.txt")
    raw_y = translator.data.load_corpus("../datasets/raw/train_ger_y.txt")
    m = TranslationModel(config)
    m.fit(raw_x, raw_y)

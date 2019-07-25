from tensorflow.keras.layers import Input, Embedding, Bidirectional, Attention, Dense, CuDNNGRU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import Model
import translator.data
import numpy as np


def create_default_model(settings, inputs_vocabulary_size, outputs_vocabulary_size):
    inputs = Input((settings['encoder_inputs'], ), name="encoder_inputs")
    embedding = Embedding(inputs_vocabulary_size + 1, settings['embedding_size'])(inputs)
    encoder = Bidirectional(CuDNNGRU(settings['units'], name="encoder", return_sequences=True))(embedding)
    decoder = CuDNNGRU(2 * settings['units'], name='decoder', return_sequences=True)(encoder)
    decoder_attention = Attention(name="decoder_attention")([encoder, decoder])
    outputs = Dense(outputs_vocabulary_size, activation='softmax', name="vocabulary")(decoder_attention)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(settings['optimizer'], settings['loss'], settings['metrics'])
    return model
    

class TranslationModel:

    def __init__(self, settings):
        self.x_vocabulary = None
        self.y_vocabulary = None
        self.model = None
        self.settings = settings

    def fit(self, x_corpus, y_corpus, epochs=1, validation_x=None, validation_y=None, callbacks=None):
        self.x_vocabulary, self.y_vocabulary = (translator.data.make_vocabulary(corpus, self.settings) for corpus in [x_corpus, y_corpus])
        if self.x_vocabulary is None or self.y_vocabulary is None:
            raise ValueError("Error fitting model: x and y vocabulary must be initialized before or during fitting.")
        if self.model is None:
            self.model = create_default_model(self.settings, len(self.x_vocabulary), len(self.y_vocabulary))
        train_dataset = translator.data.Dataset(x_corpus, y_corpus, self.x_vocabulary, self.y_vocabulary, self.settings)
        fit_kwargs = { 'epochs': epochs, 'callbacks': callbacks }
        if validation_x is not None and validation_y is not None:
            fit_kwargs['validation_data'] = translator.data.Dataset(validation_x, validation_y, self.x_vocabulary, self.y_vocabulary, self.settings)
        self.model.fit_generator(train_dataset, **fit_kwargs)
            
    def predict(x, **kwargs):
        if isinstance(x, str):
            return self.predict([x], **kwargs)
        return self.model.predict([translator.data.process_x(sample, self.x_vocabulary, self.settings) for sample in x])

    def predict_raw(self, *args, **kwargs):
        return [translator.data.convert_raw(y, self.y_vocabulary, self.settings)
                for y in self.predict(*args, **kwargs)]
    

if __name__ == '__main__':  
    raw_x = translator.data.load_corpus("../datasets/raw/train_ger_x.txt")
    raw_y = translator.data.load_corpus("../datasets/raw/train_ger_y.txt")
    m = TranslationModel({
            "encoder_inputs": 64,
            "embedding_size": 256,
            "units": 256,
            "encoder_cells": 3,
            "decoder_cells": 3,
            "optimizer": "rmsprop",
            "loss": "categorical_crossentropy",
            "batch_size": 32,
            "case_sensitive": False,
            "vocabulary_cutoff": 50000,
            "metrics": None })
    raw_validation_x = translator.data.load_corpus("../datasets/raw/validation_ger_x.txt")
    raw_validation_y = translator.data.load_corpus("../datasets/raw/validation_ger_y.txt")
    m.fit(raw_x, raw_y, validation_x=raw_validation_x, validation_y=raw_validation_y, epochs=30,
            callbacks=[ModelCheckpoint("../models/weights-{epoch:02d}-{val_loss:.3f}.hdf5"),
                        EarlyStopping(min_delta=0.0001, patience=2),
                        ReduceLROnPlateau()])
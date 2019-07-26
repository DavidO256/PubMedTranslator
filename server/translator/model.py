from tensorflow.keras.layers import Input, Embedding, Bidirectional, Attention, Dense, RNN, CuDNNLSTM, LSTMCell, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import model_from_json
from tensorflow.keras.utils import plot_model
from tensorflow.nn import log_softmax
from tensorflow.keras import Model
from contextlib import closing
import translator.data
import numpy as np
import json
import os


def create_default_model(settings, inputs_vocabulary_size, outputs_vocabulary_size):
    inputs = Input((settings['encoder_inputs'], ), name="encoder_inputs")
    embedding = Embedding(inputs_vocabulary_size + 1, settings['embedding_size'])(inputs)
    encoder = Bidirectional(CuDNNLSTM(settings['units'], name="encoder", return_sequences=True))(embedding)
    decoder = RNN([LSTMCell(2 * settings['units'], name='decoder_cell_%d' % i)
                                for i in range(settings['decoder_cells'])], return_sequences=True, name="decoder")(encoder)
    decoder_attention = Attention(name="decoder_attention")([encoder, decoder])
    link = Concatenate(name="link")([embedding, decoder, encoder])
    outputs = Dense(outputs_vocabulary_size, activation=log_softmax, name="vocabulary")(link)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(settings['optimizer'], settings['loss'], settings['metrics'])
    return model
    

class TranslationModel:

    def __init__(self, settings):
        self.settings = settings
        self.x_vocabulary = None
        self.y_vocabulary = None
        self.model = None

    def fit(self, x_corpus, y_corpus, epochs=1, validation_x=None, validation_y=None, callbacks=None):
        self.x_vocabulary, self.y_vocabulary = (translator.data.make_vocabulary(corpus, self.settings) for corpus in [x_corpus, y_corpus])
        if self.x_vocabulary is None or self.y_vocabulary is None:
            raise ValueError("Error fitting model: x and y vocabulary must be initialized before or during fitting.")
        if self.model is None:
            self.model = create_default_model(self.settings, len(self.x_vocabulary), len(self.y_vocabulary))
            plot_model(self.model, "model.png", show_shapes=True)
        train_dataset = translator.data.Dataset(x_corpus, y_corpus, self.x_vocabulary, self.y_vocabulary, self.settings)
        fit_kwargs = { 'epochs': epochs, 'callbacks': callbacks }
        if validation_x is not None and validation_y is not None:
            fit_kwargs['validation_data'] = translator.data.Dataset(validation_x, validation_y, self.x_vocabulary, self.y_vocabulary, self.settings)
        self.model.fit_generator(train_dataset, **fit_kwargs)

    def predict(self, x, **kwargs):
        if isinstance(x, str):
            return self.predict([x], **kwargs)
        inputs = np.asarray([translator.data.process_x(sample, self.x_vocabulary, self.settings['encoder_inputs']) for sample in x])
        for i in inputs:
            print(np.shape(i))
        return self.model.predict(inputs)

    def predict_raw(self, *args, **kwargs):
        return [translator.data.convert_raw(y, self.y_vocabulary, self.settings)
                for y in self.predict(*args, **kwargs)]

    def save(self, directory):
        if self.model is None or self.x_vocabulary is None or self.y_vocabulary is None:
            raise ValueError("Unable to save. Run fit or ensure the model was properly loaded.")
        with closing(open(os.path.join(directory, "x_vocabulary.json"), 'w')) as f:
            json.dump(self.x_vocabulary, f)
        with closing(open(os.path.join(directory, "y_vocabulary.json"), 'w')) as f:
            json.dump(self.y_vocabulary, f)
        with closing(open(os.path.join(directory, "settings.json"), 'w')) as f:
            json.dump(self.settings, f)
        with closing(open(os.path.join(directory, "model.json"), 'w')) as f:
            json.dump(self.model.to_json(), f)
        self.model.save_weights(os.path.join(directory, "weights.hdf5"))
    
    def load_model(self, directory):
        with closing(open(os.path.join(directory, "model.json"))) as f:
           self.model = model_from_json(str(json.load(f)), {"Attention": Attention})
           self.model.compile(self.settings['optimizer'], self.settings['loss'], self.settings['metrics'])
        with closing(open(os.path.join(directory, "x_vocabulary.json"))) as f:
            self.x_vocabulary = json.load(f)
        with closing(open(os.path.join(directory, "y_vocabulary.json"))) as f:
            self.y_vocabulary = json.load(f)
        self.model.load_weights(os.path.join(directory, "weights.hdf5"))
    
    @staticmethod
    def load(directory):
        with closing(open(os.path.join(directory, "settings.json"))) as f:
            settings = json.load(f)
        translation_model = TranslationModel(settings)
        translation_model.load_model(directory)
        return translation_model
    

if __name__ == '__main__':  
    config = {
            "encoder_inputs": 64,
            "embedding_size": 512,
            "units": 256,
            "encoder_cells": 3,
            "decoder_cells": 3,
            "optimizer": "rmsprop",
            "loss": "categorical_crossentropy",
            "batch_size": 32,
            "case_sensitive": False,
            "vocabulary_cutoff": 50000,
            "metrics": None }
    raw_x = translator.data.load_corpus("../datasets/raw/train_ger_x.txt")
    raw_y = translator.data.load_corpus("../datasets/raw/train_ger_y.txt")
    m = TranslationModel(config)
    raw_validation_x = translator.data.load_corpus("../datasets/raw/validation_ger_x.txt")
    raw_validation_y = translator.data.load_corpus("../datasets/raw/validation_ger_y.txt")
    m.fit(raw_x, raw_y, validation_x=raw_validation_x, validation_y=raw_validation_y, epochs=30,
            callbacks=[ModelCheckpoint("checkpoints/model-{epoch:02d}-{val_loss:.3f}.hdf5"),
                        EarlyStopping(min_delta=0.0001, patience=2),
                        ReduceLROnPlateau()])
    m.save("models/model")
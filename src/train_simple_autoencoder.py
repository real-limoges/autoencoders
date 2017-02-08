from keras.layers import Input, Dense, Activation
from keras.models import Model
from keras.datasets import mnist
from keras.callbacks import TensorBoard


import numpy as np

ENC_DIM = 32
DEC_DIM = 784

def arrange_data():
    '''
    Loads and arranges data
    '''
    
    (x_train, _), (x_test, _) = mnist.load_data()
    
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    return x_train, x_test


def build_model():

    input = Input(shape=(DEC_DIM,))
    encode = Dense(ENC_DIM, activation='relu')(input)
    decode = Dense(DEC_DIM, activation='sigmoid')(encode)
    autoencoder = Model(input=input, output=decode)

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy',
                        metrics = ['accuracy', 'precision', 'recall'])

    return autoencoder


if __name__ == '__main__':

    model = build_model()
    x_train, x_test = arrange_data()

    model.fit(x_train, x_train,
              nb_epoch = 50,
              batch_size = 128,
              shuffle = True,
              validation_data = (x_test, x_test),
              callbacks=[TensorBoard(log_dir='/tmp/simple_autoencoder')])

    model.save('../data/autoencoder_weights.h5')

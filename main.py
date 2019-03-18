import os
import pickle

import numpy as np

import segmentation
import cnn
import soundfile
from tensorflow import keras
# from keras.optimizers import Adam
from IPython.display import Audio as audio_playback_widget
import matplotlib.pyplot as plt

seg = segmentation.Segmentation()
cnn = cnn.Network()
prefix = 'num'

n_epochs = 20

def preprocessing():
    f = './data/raw-from-phone.wav'

    samples, sample_rate = soundfile.read(f)

    seg.show_waveform(samples, sample_rate)

    # audio_playback_widget(f) unable because IPython

    f = './data/num_phone_en-UK_m_Martin00.wav'

    seg.show_spectogram(f)

    seg.split_combined_file_into_wavs('./data/num_Bing_en-UK_f_Susan.wav')
    seg.split_all_combined_files_into_wavs()

    # stamp = seg.wav_to_stamp('num','six','phone_en-UK-m-Martin00.wav')

    seg.create_dataset_from_folders('num')

    seg.create_dataset_from_adhoc_wavs('num' + '-test')


def network():
    network = cnn.build_network()

    loss_function = 'categorical_crossentropy'
    optimizer = keras.optimizers.Adam(lr = 0.001)

    network.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])

    dataset = pickle.load(open(os.path.join('data', prefix + '.pkl'),'rb'))


    train_indices = [i for i, r in enumerate(dataset['rand']) if r <= 0.9]
    check_indices = [i for i, r in enumerate(dataset['rand']) if r > 0.9]


    ds_train = cnn.make_dataset(dataset, train_indices,n_epochs = cnn.n_epochs, batch_size=cnn.batch_size, seed=100)  # shuffles...
    ds_check = cnn.make_dataset(dataset, check_indices,n_epochs=cnn.n_epochs, batch_size=1)

    spe = len(train_indices) // cnn.batch_size

    network.fit(ds_train, steps_per_epoch=spe, epochs=cnn.n_epochs,
              validation_data=ds_check, validation_steps=len(check_indices),
              verbose=1)

    score = network.evaluate(ds_check,steps=len(check_indices),verbose=1)
    print("Score = ", score)

def main():
    preprocessing()
    network()


    

if __name__ == "__main__":
    main()

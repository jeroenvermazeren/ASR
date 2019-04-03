import os
import pickle

import numpy as np

import segmentation
import soundfile
from tensorflow import keras
# from keras.optimizers import Adam
from IPython.display import Audio as audio_playback_widget
import matplotlib.pyplot as plt

from cnn.network import Network
from segmentation.segmetation import Segmentation


class Main():
    def __init__(self, segmentation, network):
        self.segmentation = segmentation
        self.network = network
        self.prefix = 'num'

        self.n_epochs = 20

    def preprocessing(self):
        f = './data/raw-from-phone.wav'

        samples, sample_rate = soundfile.read(f)

        self.segmentation.show_waveform(samples, sample_rate)

        # audio_playback_widget(f) unable because IPython

        f = './data/num_phone_en-UK_m_Martin00.wav'

        self.segmentation.show_spectogram(f)

        self.segmentation.split_combined_file_into_wavs('./data/num_Bing_en-UK_f_Susan.wav')
        self.segmentation.split_all_combined_files_into_wavs()

        # stamp = seg.wav_to_stamp('num','six','phone_en-UK-m-Martin00.wav')

        self.segmentation.create_dataset_from_folders('num')

        self.segmentation.create_dataset_from_adhoc_wavs('num' + '-test')


    def run_network(self):
        network = self.network.build_network()

        loss_function = 'categorical_crossentropy'
        optimizer = keras.optimizers.Adam(lr = 0.001)

        network.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])

        dataset = pickle.load(open(os.path.join('data', self.prefix + '.pkl'),'rb'))


        train_indices = [i for i, r in enumerate(dataset['rand']) if r <= 0.9]
        check_indices = [i for i, r in enumerate(dataset['rand']) if r > 0.9]


        ds_train = self.network.make_dataset(dataset, train_indices,n_epochs = self.network.n_epochs, batch_size=self.network.batch_size, seed=100)  # shuffles...
        ds_check = self.network.make_dataset(dataset, check_indices,n_epochs=self.network.n_epochs, batch_size=1)

        spe = len(train_indices) // self.network.batch_size

        network.fit(ds_train, steps_per_epoch=spe, epochs=self.network.n_epochs,
                  validation_data=ds_check, validation_steps=len(check_indices),
                  verbose=1)

        score = network.evaluate(ds_check,steps=len(check_indices),verbose=1)
        print("Score = ", score)

    def main(self):
        self.preprocessing()
        self.run_network()




if __name__ == "__main__":
    main = Main(Segmentation(), Network())
    main.main()

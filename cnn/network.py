from keras import utils
import numpy as np
# from keras import Model
# from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout

class Network:
    def __init__(self, n_classes=10, batch_size = 20, n_epochs = 20, input_shape = (64,32,1)):
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.input_shape = input_shape
        self.network = None

    def build_network(self):
        ## DEFINE THE ABOVE DESCRIBED MODEL HERE
        x_in = keras.layers.Input(shape=(64,32,1))

        ### Your code starts here
        x = keras.layers.Conv2D(16, kernel_size=(5, 5), activation='relu')(x_in)
        x = keras.layers.MaxPooling2D(pool_size=(2,2))(x)

        x = keras.layers.Conv2D(16, kernel_size=(5, 5), activation='relu')(x)
        x = keras.layers.MaxPooling2D(pool_size=(2,2))(x)

        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(32, activation='relu')(x)
        x = keras.layers.Dropout(0.5)(x)

        x = keras.layers.Dense(self.n_classes, activation='softmax')(x)

        x_out = x  # this is the output fo your network, used later on in this notebook
        self.network = keras.Model(inputs=x_in, outputs=x_out)

        return self.network


    # Create Dataset iterator

    def make_dataset(self, data, indices, seed=None, n_epochs=20, batch_size=20):
        # If seed is defined, this will shuffle data into batches

        # Get the data into tensorflow
        stamps = np.array(data['stamp'])[indices]
        print("stamps.shape:", stamps.shape)
        # Ensure that the stamps are 'float32' in [0,1] and have the channel=1
        stamps_with_channel = np.expand_dims(stamps / 255.0, -1)

        labels = np.array(data['label'])[indices]
        print("labels.shape:", labels.shape)
        labels_one_hot = utils.to_categorical(labels, self.n_classes)

        all_images = tf.constant(stamps_with_channel, shape=stamps_with_channel.shape, dtype=tf.float32)
        all_labels = tf.constant(labels_one_hot, shape=labels_one_hot.shape, verify_shape=True)

        ds = tf.data.Dataset.from_tensor_slices((all_images, all_labels))
        if seed is not None:
            ds = ds.shuffle(batch_size * 4)

        ds = ds.repeat(n_epochs).batch(batch_size)

        return ds

    def get_predictions_for_dataset(self, data, network):
        n_points = len(data['stamp'])
        ds = self.make_dataset(data, range(n_points), n_epochs=1, batch_size=1)

        pred_arr = network.predict(ds, steps=n_points, verbose=0)
        # print("This is an array of predictions, each with n_classes of probs:\n",pred_arr)  # This is an array of predictions, each with n_classes of probs

        predictions = [dict(classes=i, probabilities=p, logits=np.log(p + 1e-20))
                       for i, p in enumerate(pred_arr)]

        for i, p in enumerate(predictions):
            label = int(data['label'][i])
            if label >= 0:
                p['word'] = data['words'][label]
            else:
                p['word'] = data['words'][i]
            p['label'] = label

        return predictions

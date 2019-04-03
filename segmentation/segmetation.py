import os
import pickle
import re
import PIL
import matplotlib.pyplot as plt
import numpy as np
import python_speech_features
import scipy
import scipy.misc
import soundfile
from IPython.display import Audio as audio_playback_widget


class Segmentation:

    def __init__(self):
        self.sample_window_step = 0.01  # in seconds (10ms)
        self.f = './data/raw-from-phone.wav'  # File location

        self.sentences = dict(
            num=self.words("zero one two three four five six seven eight nine."),

            animals=self.words("cat dog fox bird."),

            # https://www.quora.com/Is-there-a-text-that-covers-the-entire-English-phonetic-range/
            qbf=self.words("That quick beige fox jumped in the air over each thin dog.  " +
                           "Look out, I shout, for he's foiled you again, creating chaos."),
            shy=self.words("Are those shy Eurasian footwear, cowboy chaps, " +
                           "or jolly earthmoving headgear?"),
            ate=self.words("The hungry purple dinosaur ate the kind, zingy fox, the jabbering crab, " +
                           "and the mad whale and started vending and quacking."),
            suz=self.words("With tenure, Suzie'd have all the more leisure for yachting, " +
                           "but her publications are no good."),
            tbh=self.words("Shaw, those twelve beige hooks are joined if I patch a young, gooey mouth."),

            #  https://en.wikipedia.org/wiki/The_North_Wind_and_the_Sun          #594
            #  http://videoweb.nie.edu.sg/phonetic/courses/aae103-web/wolf.html  #1111
        )

    # Read in the original file

    def show_waveform(self, sound, sample_rate):
        n_samples = sound.shape[0]

        plt.figure(figsize=(12, 2))
        plt.plot(np.arange(0.0, n_samples) / sample_rate, sound)
        plt.xticks(np.arange(0.0, n_samples / sample_rate, 0.5), rotation=90)

        plt.grid(True)

        plt.show()

    def play_audio(self, f):
        samples, sample_rate = soundfile.read(f)
        self.show_waveform(samples, sample_rate)
        audio_playback_widget(f)

    def get_sample_features(self, samples, sample_rate):
        # sample_feat = python_speech_features.mfcc(samples, sample_rate, numcep=13, nfilt=26, appendEnergy=True)
        # sample_feat = python_speech_features.mfcc(samples, sample_rate, numcep=28, nfilt=56, appendEnergy=True)

        # sample_feat, e = python_speech_features.fbank(samples,samplerate=sample_rate,
        #      winlen=0.025,winstep=0.01,nfilt=26,nfft=512,
        #      lowfreq=0,highfreq=None,preemph=0.97, winfunc=lambda x:np.ones((x,)))

        features, energy = python_speech_features.fbank(samples, samplerate=sample_rate,
                                                        winlen=0.025, winstep=self.sample_window_step,
                                                        nfilt=32, nfft=512,
                                                        lowfreq=0, highfreq=None, preemph=0.25,
                                                        winfunc=lambda x: np.hamming(x))
        return features, energy

    def get_sample_isolated_words(self, energy, plot=False):
        log_e = np.log(energy)
        if plot: plt.plot(log_e - 5)

        # log_e = smooth(log_e)
        # if plot: plt.plot(log_e)

        log_e_hurdle = (log_e.max() - log_e.min()) * 0.5 + log_e.min()

        log_e_crop = np.where(log_e > log_e_hurdle, 1.0, 0.0)
        if plot: plt.plot(log_e_crop * 25 - 2.5)

        # By smoothing, and applying a very low hurdle, we expand the crop area safely
        log_e_crop_expanded = np.where(self.smooth(log_e_crop, ) > 0.01, 1.0, 0.0)
        if plot: plt.plot(log_e_crop_expanded * 30 - 5)

        return self.contiguous_regions(log_e_crop_expanded > 0.5)

    def spectrogram(self, wav_filepath):
        samples, sample_rate = soundfile.read(wav_filepath)

        # Original code from :
        # https://mail.python.org/pipermail/chicago/2010-December/007314.html

        # Rescale so that max/min are ~ +/- 1 around 0
        data_av = np.mean(samples)
        data_max = np.max(np.absolute(samples - data_av))
        sound_data = (samples - data_av) / data_max

        ## Parameters: 10ms step, 30ms window
        nstep = int(sample_rate * 0.01)
        nwin = int(sample_rate * 0.03)
        nfft = 2 * int(nwin / 2)

        window = np.hamming(nwin)

        # will take windows x[n1:n2].  generate and loop over
        # n2 such that all frames fit within the waveform
        nn = range(nwin, len(sound_data), nstep)

        X = np.zeros((len(nn), nfft // 2))

        for i, n in enumerate(nn):
            segment = sound_data[n - nwin:n]
            z = np.fft.fft(window * segment, nfft)
            X[i, :] = np.log(np.absolute(z[:nfft // 2]))

        return X

    # This is a function that smooths a time-series
    #   which enables us to segment the input into words by looking at the 'energy' profile
    def smooth(self, x, window_len=31):  # , window='hanning'
        # http://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
        # s = np.r_[ x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
        s = np.r_[np.zeros(((window_len - 1) // 2,)), x, np.zeros(((window_len - 1) // 2,))]
        w = np.hamming(window_len)
        return np.convolve(w / w.sum(), s, mode='valid')  # [window_len-1 : -(window_len-1) ]

    # http://stackoverflow.com/questions/4494404/find-large-number-of-consecutive-values-fulfilling-condition-in-a-numpy-array
    def contiguous_regions(self, condition):
        idx = []
        i = 0
        while i < len(condition):
            x1 = i + condition[i:].argmax()
            try:
                x2 = x1 + condition[x1:].argmin()
            except:
                x2 = x1 + 1
            if x1 == x2:
                if condition[x1] == True:
                    x2 = len(condition)
                else:
                    break
            idx.append([x1, x2])
            i = x2
        return idx

    def words(self, s):
        remove_punc = re.compile('[\,\.\?\!]')
        squash_spaces = re.compile('\s+')

        s = remove_punc.sub(' ', s)
        s = squash_spaces.sub(' ', s)
        return s.strip().lower()

    def for_msft(prefixes, sentences):  # comma separated
        return ' '.join([sentences[a] for a in prefixes.split(',')]).replace(' ', '\n')

    # Convert a given (isolated word) WAV into a 'stamp' - using a helper function

    def samples_to_stamp(self, samples, sample_rate):
        sample_feat, energy = self.get_sample_features(samples, sample_rate)

        data = np.log(sample_feat)

        # Now normalize each vertical slice so that the minimum energy is ==0
        data_mins = np.min(data, axis=1)
        data_min0 = data - data_mins[:, np.newaxis]

        # Force the data into the 'stamp size' as an image (implicit range normalization occurs)
        stamp = scipy.misc.imresize(data_min0, (64, 32), 'bilinear')

        # https://github.com/scipy/scipy/issues/4458 :: The stamps are stored as uint8...
        return stamp

    def wav_to_stamp(self, prefix, word, wav):
        samples, sample_rate = soundfile.read(os.path.join('data', prefix, word, wav))
        return self.samples_to_stamp(samples, sample_rate)

    # Now do something similar for 'test files', create a dataset for all the audio files in the given folder

    def create_dataset_from_adhoc_wavs(self, prefix, save_as='.pkl', seed=13):
        stamps, labels, words = [], [], []

        for audio_file in sorted(os.listdir(os.path.join('data', prefix))):
            filename_stub, ext = os.path.splitext(audio_file)
            if not (ext == '.wav' or ext == '.ogg'): continue

            samples, sample_rate = soundfile.read(os.path.join('data', prefix, audio_file))
            print("Sample_rate = ", sample_rate)
            print("samples = ", samples)
            sample_feat, energy = self.get_sample_features(samples, sample_rate)
            word_ranges = self.get_sample_isolated_words(energy, plot=False)
            print("word_ranges = ", word_ranges)
            for i, wr in enumerate(word_ranges):
                wr = word_ranges[i]
                fac = int(self.sample_window_step * sample_rate)
                segment = samples[wr[0] * fac:wr[1] * fac]

                stamp = self.samples_to_stamp(segment, sample_rate)

                print("Adding : %s #%2d : (%d,%d)" % (filename_stub, i, wr[0], wr[1],))
                stamps.append(stamp)
                labels.append(-1)
                words.append("%s_%d" % (filename_stub, i))

        np.random.seed(seed)
        data_dictionary = dict(
            stamp=stamps, label=labels,
            rand=np.random.rand(len(labels)),
            words=words,
        )
        ds_file = os.path.join('data', prefix + save_as)
        pickle.dump(data_dictionary, open(ds_file, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        print("Created dataset : %s" % (ds_file,))



    def show_spectogram(self, f):
        X = self.spectrogram(f)
        print("X.shape=", X.shape)

        # Y = np.std(X, axis=1)
        Y = np.max(X, axis=1)
        Y_min = np.min(Y)
        Y_range = Y.max() - Y_min
        Y = (Y - Y_min) / Y_range

        print("Y.shape=", Y.shape)

        Y_crop = np.where(Y > 0.50, 1.0, 0.0)

        # Apply some smoothing
        Y_crop = self.smooth(Y_crop)

        Y_crop = np.where(Y_crop > 0.01, 1.0, 0.0)
        print("Y_crop.shape=", Y_crop.shape)

        plt.figure(figsize=(12, 3))
        plt.imshow(X.T, interpolation='nearest', origin='lower', aspect='auto')
        plt.title(f)
        plt.xlim(xmin=0)
        plt.ylim(ymin=0)

        plt.plot(Y * X.shape[1])

        plt.plot(Y_crop * X.shape[1])

        plt.show()
        # Y.min(), Y.max()
        # X[100,:]
        print(np.argmin(X) / 248, np.argmax(X) / 248)


    def split_combined_file_into_wavs(self,f, prefix='num'):
        # f ~ './data/num_Bing_en-UK_f_Susan.wav'
        f_base_orig = os.path.basename(f)
        if not f_base_orig.startswith(prefix + "_"):
            print("Wrong prefix for '%s'" % (f_base_orig,))
            return

        # Here's the new filename (directory to be calculated per-word)
        f_base = os.path.splitext(f_base_orig)[0][len(prefix) + 1:] + '.wav'

        samples, sample_rate = soundfile.read(f)
        sample_feat, energy = self.get_sample_features(samples, sample_rate)
        word_ranges = self.get_sample_isolated_words(energy, plot=False)
        # print(word_ranges)

        words = self.sentences[prefix].split(' ')
        if len(word_ranges) != len(words):
            print("Found %d segments, rather than %d, in '%s'" % (len(word_ranges), len(words), f,))
            return

        for i, word in enumerate(words):
            word_path = os.path.join('data', prefix, word)
            os.makedirs(word_path, exist_ok=True)

            wr = word_ranges[i]
            fac = int(self.sample_window_step * sample_rate)
            soundfile.write(os.path.join(word_path, f_base), samples[wr[0] * fac:wr[1] * fac], samplerate=sample_rate)

    def split_all_combined_files_into_wavs(self, prefix='num'):
        for audio_file in sorted(os.listdir('./data')):
            filename_stub, ext = os.path.splitext(audio_file)
            if not (ext == '.wav' or ext == '.ogg'): continue
            if not filename_stub.startswith(prefix + '_'): continue

            # print("Splitting %s" % (audio_file,))
            self.split_combined_file_into_wavs('./data/' + audio_file, prefix=prefix)


    def create_dataset_from_folders(self,prefix, save_as='.pkl', seed=13):
        words = self.sentences[prefix].split(' ')
        stamps, labels = [], []

        for label_i, word in enumerate(words):
            # Find all the files for this word
            for stamp_file in os.listdir(os.path.join('data', prefix, word)):
                if not stamp_file.endswith('.wav'): continue
                # print(stamp_file)
                stamp = self.wav_to_stamp(prefix, word, stamp_file)

                stamps.append(stamp)
                labels.append(label_i)

        if save_as is None:  # Return the data directly
            return stamps, labels, words

        np.random.seed(seed)
        data_dictionary = dict(
            stamp=stamps, label=labels,
            rand=np.random.rand(len(labels)),  # This is to enable us to sample the data (based on hurdles)
            words=words,
        )
        ds_file = os.path.join('data', prefix + save_as)
        pickle.dump(data_dictionary, open(ds_file, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        print("Created dataset : %s" % (ds_file,))
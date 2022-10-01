from glob import glob
import pandas as pd
import os
import shutil
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import librosa
import soundfile as sf
from audiomentations import Compose, AddGaussianNoise, PitchShift, HighPassFilter
import cv2

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
import tensorflow_io as tfio
from keras.models import Sequential, model_from_json
from keras import layers
from keras.losses import SparseCategoricalCrossentropy
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

def sort_data(parent_dir = "", target_dir ="", meta_data = ""):
    meta_data = pd.read_csv(meta_data)
    print(meta_data.head())

    classes = meta_data['category'].unique()

    for category in classes:
        PATH = os.path.join(target_dir, category)
        if not os.path.isdir(PATH):
            os.mkdir(PATH)
        
        subset = meta_data[meta_data["category"] == category]
        files = list(subset["filename"])
        for file in files:
            file_path = os.path.join(parent_dir, file)
            destination = os.path.join(target_dir, category)
            shutil.move(file_path, destination)

class build_spectrograms(object):
    def __init__(self, audio_binary = "", force_sr = 16000) -> None:
        self.data_in = audio_binary
        self.sample_rate = force_sr
        self.label = audio_binary.split("/")[-2]

        return None

    def decode_augment_audio(self, augmenter, num_augmentations):
        
        working_data = []
        audio, sr = librosa.load(self.data_in, sr = self.sample_rate)
        working_data.append(audio)
        for i in range(0,num_augmentations,1) :
            augmented_signal = augmenter(audio, sr)
            working_data.append(augmented_signal)
        
        self.waveforms = working_data
        # Data is single channel (mono), drop the `channels` axis from the array.
        # self.waveform = tf.squeeze(audio, axis = -1)
        # print(audio.shape)
        return self

    def get_spectrograms(self, final_dim):
        '''
            No need for padding in the end
        '''
        spectros = []
        for waveform in self.waveforms:
            waveform = tf.cast(waveform, dtype=tf.float32)

            spectrogram = tfio.audio.spectrogram(
                waveform, nfft=512, window=512, stride=256)
            spectrogram = tfio.audio.melscale(
                spectrogram, rate=self.sample_rate, mels=256, fmin=0, fmax=8000)
            spectrogram = tfio.audio.dbscale(
                spectrogram, top_db=45)

            # Obtain the magnitude of the STFT - not necassary?
            # spectrogram = tf.abs(spectrogram)
            '''
            Add a `channels` dimension, so that the spectrogram can be used
            as image-like input data with convolution layers (which expect
            shape (`batch_size`, `height`, `width`, `channels`).
            '''
            spectrogram = cv2.resize(spectrogram.numpy(), final_dim, interpolation = cv2.INTER_AREA)
            spectros.append(spectrogram)
        
        
        return spectros, self.label

def plot_spectrogram(spectrogram):
    if len(spectrogram.shape) > 2:
        assert len(spectrogram.shape) == 3
        spectrogram = np.squeeze(spectrogram, axis=-1)
    # Convert the frequencies to log scale and transpose, so that the time is
    # represented on the x-axis (columns).
    # Add an epsilon to avoid taking a log of zero.
    log_spec = (spectrogram.T + np.finfo(float).eps)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    plt.subplots(1, figsize=(12, 8))
    plt.pcolormesh(X, Y, log_spec)
    plt.title('Centred dbScale Mel-Spectrogram')
    plt.tight_layout()
    plt.show()

class customModel (object):
    def __init__(self, set_height, set_width, set_colour_depth) -> None:
        self.height = set_height
        self.width = set_width
        self.depth = set_colour_depth

    def buildModel (self, num_outputs):
        self.model = Sequential(
            [
                layers.Input(shape=(self.height , self.width , self.depth)),
            
                #Block 1
                layers.Conv2D(filters = 64, kernel_size = (3,3), padding='same' , activation='relu'),
                layers.Conv2D(filters = 64, kernel_size = (3,3), padding='same' , activation='relu'),
                layers.MaxPool2D(pool_size=(2,2) , strides=(2,2) , padding='same'),
                layers.BatchNormalization(),
                
                #Block 2
                layers.Conv2D(filters = 128, kernel_size = (3,3), padding='same' , activation='relu'),
                layers.Conv2D(filters = 128, kernel_size = (3,3), padding='same' , activation='relu'), 
                layers.MaxPool2D(pool_size=(2,2) , strides=(2,2) , padding='same'),
                layers.BatchNormalization(),
                
                #Block 3
                layers.Conv2D(filters = 256, kernel_size = (3,3), padding='same' , activation='relu'),
                layers.Conv2D(filters = 256, kernel_size = (3,3), padding='same' , activation='relu'),
                layers.Conv2D(filters = 256, kernel_size = (3,3), padding='same' , activation='relu'),
                layers.MaxPool2D(pool_size=(2,2) , strides=(2,2) , padding='same'),
                layers.BatchNormalization(),
                
                #Block 4
                layers.Conv2D(filters = 512, kernel_size = (3,3), padding='same' , activation='relu'),
                layers.Conv2D(filters = 512, kernel_size = (3,3), padding='same' , activation='relu'),
                layers.Conv2D(filters = 512, kernel_size = (3,3), padding='same' , activation='relu'),
                layers.MaxPool2D(pool_size=(2,2) , strides=(2,2) , padding='same'),
                layers.BatchNormalization(),
                
                # #Block 5
                # layers.Conv2D(filters = 512, kernel_size = (3,3), padding='same' , activation='relu'),
                # layers.Conv2D(filters = 512, kernel_size = (3,3), padding='same' , activation='relu'),
                # layers.Conv2D(filters = 512, kernel_size = (3,3), padding='same' , activation='relu'),
                # layers.MaxPool2D(pool_size=(2,2) , strides=(2,2) , padding='same'),
                # layers.BatchNormalization(),
                
                #Block 6
                layers.Flatten(),
                layers.Dense(units = 512 , activation='relu'),
                layers.Dropout(rate = 0.2),
                layers.Dense(units = 256 , activation='relu'),
                layers.Dropout(rate = 0.2),
                layers.Dense(units = 128 , activation='relu'),
                layers.Dropout(rate = 0.2),
                layers.Dense(units = num_outputs , activation='softmax')
            ]
        )
        return self
    
    def compileModel (self, optimiser, loss_metric):
        self.model.compile(
            metrics=['Accuracy'],
            optimizer=optimiser,
            loss = loss_metric
        )
        self.model.summary()
        return self.model
    
def visualise_performance (history):
    fig, axs = plt.subplots(2)
    axs[0].plot(history.history['loss'], label = "Loss")
    axs[0].plot(history.history['val_loss'], label = "Validation Loss")
    axs[0].legend()
    axs[1].plot(history.history['Accuracy'], label = "Accuracy")
    axs[1].plot(history.history['val_Accuracy'], label = "Validation Accuracy")
    axs[1].legend()
    plt.tight_layout()
    plt.show()

def save_model(model_in, save_dir = "", save_name = ""):
    model_json = model_in.to_json()
    with open(f'{os.path.join(save_dir,save_name)}.json', "w") as json_file:
        json_file.write(model_json)
    json_file.close()
    # serialize weights to HDF5
    model_in.save_weights(f'{os.path.join(save_dir,save_name)}.hdf5')
    print("Saved model to disk")

def load_model(save_dir = "", model_name = ""):
    print("Loading Precomputed Model")
    json_file = open(f'{os.path.join(save_dir,model_name)}.json', 'r').read()
    model = model_from_json(json_file)
    # load weights into new model
    model.load_weights(f'{os.path.join(save_dir,model_name)}.hdf5')
    print("Loaded model from disk")
    model.summary()
    return model

def model_accuracy(model, data, labels_dict):
    model.evaluate(data)
    y_pred = model.predict(data)
    y_pred = tf.argmax(y_pred, axis=1).numpy()

    y_true = np.concatenate([y for x, y in data], axis=0)

    labels = np.array(list(labels_dict.keys()))  

    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(confusion_mtx,
                xticklabels=labels,
                yticklabels=labels,
                annot=True, fmt='g')

    for text in ax.texts:
        text.set_size(6)
        if text.get_text() == '0':
            text.set_alpha(0)

    plt.xlabel('Prediction')
    plt.ylabel('Real Label')
    plt.show()

# def run_inference(model, file = "", model_width = 0, model_height = 0, model_depth = 3):
#     INDICES = {'buildings': 0, 'forest': 1, 'glacier': 2, 'mountain': 3, 'sea': 4, 'street': 5}

#     image = imread(file)
#     image = resize(
#         image, (model_width, model_height), interpolation = INTER_AREA).reshape(
#             (1, model_width, model_height, model_depth)
#         )
#     image = image/255
#     prediction = model.predict(image).flatten()
#     prediction_int = tf.argmax(prediction)

#     named_output = list(INDICES.keys())[prediction_int]
#     probability = prediction[prediction_int]

#     return named_output, probability
    
if __name__ == "__main__":
    pass
    '''
        Sort Data
    '''
    # sort_data(parent_dir = "./audio", target_dir = "./raw_data", meta_data="./esc50.csv")
    # '''
    #     File Augmentation - takes a while
    # '''
    
    '''
        Check audio file durations - all 5 seconds
    '''
    # durations = []
    # for dir in dirs:
    #     files = glob(os.path.join(dir, "*"))
    #     for file in files:
    #         durations.append(librosa.get_duration(filename=file, sr=None))
    # durations = np.array(durations)
    # print(durations)
    
# augment = True #@param
# dirs = glob("./raw_data/*")
# NUM_AUGMENTATIONS = 3
# if augment:
#     augment_raw_audio = Compose(
#         [
#             AddGaussianNoise(min_amplitude=0.01, max_amplitude=0.025, p=0.5),
#             PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
#             # HighPassFilter(min_cutoff_freq=2000, max_cutoff_freq=4000, p=0.25)
#         ]
#     )
#     for dir in tqdm(dirs):
#         files = glob(os.path.join(dir, "*"))
#         for file in files:
#             if "augmented" in file:
#                 os.remove(file)
#     # for dir in tqdm(dirs):
#     #     files = glob(os.path.join(dir, "*"))
#     #     for file in files:
#     #         # print(file)
#     #         ext.augment_data(
#     #             augmenter=augment_raw_audio,
#     #             file=file,
#     #             num_augmentations=NUM_AUGMENTATIONS,
#     #             out_dir=dir   
#     #         )

# def augment_data(augmenter, file = "", num_augmentations = 1, out_dir = ""):
#     signal, sr = librosa.load(file)

#     for i in range(0,num_augmentations,1) :
#         augmented_signal = augmenter(signal, sr)

#         file_name = file.split("/")[-1]
#         # print(file_name)
#         file_out = os.path.join(
#             out_dir, f'augmented-{i}-{file_name}'
#         )
#         sf.write(file_out, augmented_signal, sr)
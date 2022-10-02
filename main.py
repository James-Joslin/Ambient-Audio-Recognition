import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import librosa
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from keras.losses import SparseCategoricalCrossentropy
import tensorflow_io as tfio
import cv2

import external as ext

if __name__ == "__main__":
    # def run_inference(model, file = "", model_width = 0, model_height = 0, model_depth = 3):
    HEIGHT = 128
    WIDTH = 128
    DEPTH = 1
    RATE = 16000
    INDICES = {
        'airplane': 0, 'breathing': 1, 'brushing_teeth': 2, 'can_opening': 3, 'car_horn': 4,
        'cat': 5, 'chainsaw': 6, 'chirping_birds': 7, 'church_bells': 8, 'clapping': 9,
        'clock_alarm': 10, 'clock_tick': 11, 'coughing': 12, 'cow': 13, 'crackling_fire': 14,
        'crickets': 15, 'crow': 16, 'crying_baby': 17, 'dog': 18, 'door_wood_creaks': 19,
        'door_wood_knock': 20, 'drinking_sipping': 21, 'engine': 22, 'fireworks': 23, 'footsteps': 24,
        'frog': 25, 'glass_breaking': 26, 'hand_saw': 27, 'helicopter': 28, 'hen': 29, 'insects': 30,
        'keyboard_typing': 31, 'laughing': 32, 'mouse_click': 33, 'pig': 34, 'pouring_water': 35, 'rain': 36,
        'rooster': 37, 'sea_waves': 38, 'sheep': 39, 'siren': 40, 'sneezing': 41, 'snoring': 42, 'thunderstorm': 43,
        'toilet_flush': 44, 'train': 45, 'vacuum_cleaner': 46, 'washing_machine': 47, 'water_drops': 48, 'wind': 49
    }
    scaler = MinMaxScaler()

    model = ext.load_model(
        save_dir = "./saved_models/", model_name = "Multi-Class_AudioRecognition",
        show_summary = False
    )
    model.compile(
        optimizer = "Nadam",
        loss = SparseCategoricalCrossentropy(
            from_logits=False),
        metrics='Accuracy'
    )

    # Random file found on the internet
    waveform, _ = librosa.load("./527664__straget__thunder.wav", sr = RATE)
    position = tfio.audio.trim(waveform, axis=0, epsilon=0.3)
    start = position[0]
    stop = position[1]

    # waveform = waveform[start:stop]

    file_duration = len(waveform)
    required_duration = 5 * RATE

    if file_duration < required_duration:
        zero_padding = tf.zeros(
            required_duration - file_duration,
            dtype=tf.float32)
        zero_padding = zero_padding[0:int(len(zero_padding)/2)]
        
        # Cast the waveform tensors' dtype to float32.
        waveform = tf.cast(waveform, dtype=tf.float32)
        '''
            Concatenate the waveform with `zero_padding`, which ensures all audio
            clips are of the same length.
        '''
        waveform = tf.concat([zero_padding, waveform, zero_padding], 0)

    elif file_duration > required_duration:
        print(len(waveform))

    else:
        pass

    spectrogram = tfio.audio.spectrogram(
            waveform, nfft=512, window=512, stride=256)
    spectrogram = tfio.audio.melscale(
        spectrogram, rate=RATE, mels=256, fmin=0, fmax=8000)
    spectrogram = tfio.audio.dbscale(
        spectrogram, top_db=45)

    spectrogram = cv2.resize(spectrogram.numpy(), (WIDTH, HEIGHT), interpolation = cv2.INTER_AREA)

    spectrogram = scaler.fit_transform(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]
    spectrogram = spectrogram.reshape((1, spectrogram.shape[0], spectrogram.shape[1], spectrogram.shape[2]))
    print(spectrogram.shape)

    prediction = model.predict(spectrogram).flatten()
    prediction_int = tf.argmax(prediction)

    named_output = list(INDICES.keys())[prediction_int]
    probability = prediction[prediction_int]

    print(f'Prediction: {named_output}\nProbability: {probability}')

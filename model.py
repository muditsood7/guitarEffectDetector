import math

import librosa
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from werkzeug.utils import secure_filename
import os

SAMPLE_RATE=22050
DURATION=5
SAMPLES_PER_TRACK=SAMPLE_RATE*DURATION

model = tf.keras.models.load_model('C:/Users/mudit/PycharmProjects/guitarEffectDetector/cnnModel.keras')

def preprocess_file(file, n_MFCC=13, n_fft=2048, hop_length=512, num_segments=5):

    # dictionary for storing data
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": []
    }

    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)

    signal, sr = librosa.load(file, sr=SAMPLE_RATE)

    # process segments, extracting the MFCCs and storing data

    for s in range(num_segments):
        start_sample = num_samples_per_segment * s  # s=0 -> 0
        finish_sample = start_sample + num_samples_per_segment

        mfcc = librosa.feature.mfcc(y=signal[start_sample:finish_sample],
                                    sr=sr,
                                    n_fft=n_fft,
                                    n_mfcc=n_MFCC,
                                    hop_length=hop_length)
        mfcc = mfcc.T

        # store mfcc for segment if it has the expected length
        if len(mfcc) == expected_num_mfcc_vectors_per_segment:
            data["mfcc"].append(mfcc.tolist())
            data["labels"].append(0)  # sub 1 because first iteration is for data path

            print("{}, segment:{}".format(file, s + 1))



def predict_note(file):

    # Store audio file
    filename = secure_filename(file.filename)
    file_path = os.path.join("uploads", filename)
    file.save(file_path)

    # Preprocess the audio file
    audio_data = preprocess_file(file_path)

    # Run the audio data through the TensorFlow model
    predictions = model.predict(np.expand_dims(audio_data, axis=0))

    return predictions

import os
import librosa
import math
import json

DATASET_PATH = "EGFxSet"
JSON_PATH = "Data.json"

SAMPLE_RATE=22050
DURATION=5
SAMPLES_PER_TRACK=SAMPLE_RATE*DURATION


# Function which segments each track, and stores its labels and mfccs.
def save_MFCC(dataset_path, json_path, n_MFCC=13, n_fft=2048, hop_length=512, num_segments=5):

    # dictionary for storing data
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": []
    }

    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)

    # loop through all the genres in the dataset
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure that the loop is not at the root level
        if dirpath is not dataset_path:
            # save the semantic label
            dirpath_components = dirpath.split("\\") # genre/blues => ['genre', 'blues']
            semantic_label = dirpath_components[-1]
            data["mapping"].append(semantic_label)
            print("\n Processing {}".format(semantic_label))

            # process files for a genre

            for f in filenames:

                # load audio file
                file_path = os.path.join(dirpath, f)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                # process segments, extracting the MFCCs and storing data

                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s # s=0 -> 0
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
                        data["labels"].append(i-1) # sub 1 because first iteration is for data path

                        print("{}, segment:{}".format(file_path, s+1))

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

if __name__ == "__main__":

    save_MFCC(DATASET_PATH, JSON_PATH, num_segments=2)
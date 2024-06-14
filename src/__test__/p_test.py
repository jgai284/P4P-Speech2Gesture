# What is .p file
# A .p file can be a serialized model or Python object created using the pickle module.

import pickle
from sklearn import datasets, model_selection, metrics

def main():
    # TODO: inspect the content of a .p file
    path2weight = r"save\speech2gesture\oliver\exp_105_cpk_speech2gesture_speaker_['oliver']_model_Speech2Gesture_G_note_speech2gesture_weights.p"

    with open(path2weight, 'rb') as file:
        data = pickle.load(file)

    print(data)

if __name__ == "__main__":
    main()
# P4P-Speech2Gesture

Introduction of P4P

## Dataset

[PATS](https://chahuja.com/pats/) and [Repo](https://github.com/chahuja/pats)

## Set up environment

### pycasper

Windows

1. Open cmd as admin
2. Navigate to `P4P-Speech2Gesture`
3. `mkdir ..\pycasper`
4. `git clone https://github.com/chahuja/pycasper ..\pycasper`
5. `cd src`
6. `mklink /D pycasper ..\..\pycasper\pycasper`

### Dependencies

- `pip install -r requirements.txt`

### ffmpeg

1. download [ffmpeg](https://github.com/BtbN/FFmpeg-Builds/releases)
2. unzip the packgae
3. add the directory there .exe files are stored to system path
4. check ffmpeg installation `ffmpeg --version`

## Virtual environment

### Windows

1. Create new virtual environment: `python -m venv .venv` <br>
2. Activate virtual environment: `.venv\Scripts\activate` <br>
3. Exit virtual environment: `deactivate` <br>

### Linux

1. Create new virtual environment: `python3 -m venv .venv` <br>
2. Activate virtual environment: `source .venv/bin/activate` <br>
3. Exit virtual environment: `deactivate` <br>

## Training

Windows:

```sh
python src/train.py -batch_size 32 -cpk speech2gesture -early_stopping 0 -exp 1 -fs_new '[15, 15]' -gan 1 -loss L1Loss -model Speech2Gesture_G -note speech2gesture -num_epochs 100 -overfit 0 -render 0 -save_dir save/speech2gesture/oliver -stop_thresh 3 -tb 1 -window_hop 5
```

Linux:

```sh
python src/train.py -path2data 'F:/PATSDATASET/oliver/pats/data' -path2outdata 'F:/PATSDATASET/oliver/pats/data' -batch_size 32 -cpk speech2gesture -early_stopping 0 -exp 1 -fs_new '[15, 15]' -gan 1 -input_modalities '["audio/log_mel_400"]' -loss L1Loss -modalities '["pose/data", "audio/log_mel_400"]' -model Speech2Gesture_G -note speech2gesture -num_epochs 100 -overfit 0 -render 0 -save_dir save/speech2gesture/oliver -speaker '["oliver"]' -stop_thresh 3 -tb 1 -window_hop 5
```

## Quantitative evaluation

```sh
cd src
```

```sh
python sample.py -load "D:\UoA\SOFTENG700A\P4P-Speech2Gesture\save\speech2gesture\oliver\exp_105_cpk_speech2gesture_speaker_['oliver']_model_Speech2Gesture_G_note_speech2gesture_weights.p" -path2data "F:\PATSDATASET\oliver\pats\data"
```

## Evaluation metrics

TBD

## Inference

```sh
cd src
```

```sh
python sample.py -load "D:\UoA\SOFTENG700A\P4P-Speech2Gesture\save\speech2gesture\oliver\exp_105_cpk_speech2gesture_speaker_['oliver']_model_Speech2Gesture_G_note_speech2gesture_weights.p" -sample_all_styles 20 -path2data "F:\PATSDATASET\oliver\pats\data"
```

## Rendering

```sh
cd src
```

```sh
python render.py -render 20 -load "D:\UoA\SOFTENG 700A\P4P-Speech2Gesture\save\speech2gesture\oliver\exp_105_cpk_speech2gesture_speaker_['oliver']_model_Speech2Gesture_G_note_speech2gesture_weights.p" -render_text 0 -path2data "F:\PATSDATASET\oliver\pats\data"
```

## Codebase introduction

### gan.py

- Focuses on the GAN framework, including the training loop, loss calculations, and integration of generator and discriminator for the training process

### speech2gesture.py

- Defines the specific architectures of the generator and discriminator models used within the GAN framework

## Dependencies

### scipy:

- `pip install scipy==1.10.1`

### gensim:

- TBD

### webrtcvad:

- Install [Microsoft C++ build tool](https://visualstudio.microsoft.com/visual-cpp-build-tools/) on Windows

## Bug report

### style_classifier.py - [line 20](https://github.com/jgai284/P4P-Speech2Gesture/blob/main/src/model/style_classifier.py#L20)

- Change default num of speaker from input to 25 if pretrained model (trainer.py - [line 406](https://github.com/jgai284/P4P-Speech2Gesture/blob/main/src/model/trainer.py#L406)) is activated

### dataUtils.py - [line 138](https://github.com/jgai284/P4P-Speech2Gesture/blob/main/src/data/dataUtils.py#L138)

- Update pandas data structure operation

### dataUtils.py - [line 277](https://github.com/jgai284/P4P-Speech2Gesture/blob/main/src/data/dataUtils.py#L277)

- Convert byte strings in missing_intervals to regular strings

### trainer.py - [line 62](https://github.com/jgai284/P4P-Speech2Gesture/blob/main/src/model/trainer.py#L62)

- Have to manually replace speaker arugment if pretrained model (trainer.py - [line 406](https://github.com/jgai284/P4P-Speech2Gesture/blob/main/src/model/trainer.py#L406)) is activated otherwise it would become shelly

### trainer.py - [line 406](https://github.com/jgai284/P4P-Speech2Gesture/blob/main/src/model/trainer.py#L406)

- Deactivate pretrained model since we do not have a pretrained speech2gesture model

### speech2gesture.py - [line 30](https://github.com/jgai284/P4P-Speech2Gesture/blob/main/src/model/speech2gesture.py#L30)

- Added missing field `kwargs`

### argUtils.py - [Dataset Parameters](https://github.com/jgai284/P4P-Speech2Gesture/blob/main/src/argsUtils.py) (not a bug actually)

- Updated some parameters to ensure compatibility issue with Windows, dataset storage, data type (originally designed for Linux)

## Lab workstation setup

[Instructions](GPU.md)

## Reference

- [Speech2Gesture](https://people.eecs.berkeley.edu/~shiry/speech2gesture/)<br>
- [Mix-stage](https://github.com/chahuja/mix-stage)

# P4P-Speech2Gesture

Introduction of P4P

## Dataset

PATS

## Set up environment

TBD

## Virtual environment

### Windows

Create new virtual environment: `python -m venv .venv` <br>
Activate virtual environment: `.venv\Scripts\activate` <br>
Exit virtual environment: `deactivate` <br>

### Linux

Create new virtual environment: `python3 -m venv .venv` <br>
Activate virtual environment: `source .venv/bin/activate` <br>
Exit virtual environment: `deactivate` <br>

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
python sample.py -load "D:\UoA\SOFTENG700A\P4P-Speech2Gesture\save\speech2gesture\oliver\exp_105_cpk_speech2gesture_speaker_['oliver']_model_Speech2Gesture_G_note_speech2gesture_weights.p" -path2data "F:\PATSDATASET\oliver\pats\data"
```

## Evaluation metrics

TBD

## Inference

```sh
python sample.py -load "D:\UoA\SOFTENG700A\P4P-Speech2Gesture\save\speech2gesture\oliver\exp_105_cpk_speech2gesture_speaker_['oliver']_model_Speech2Gesture_G_note_speech2gesture_weights.p" -sample_all_styles 20 -path2data "F:\PATSDATASET\oliver\pats\data"
```

## Rendering

```sh
python render.py -render 20  -load "D:\UoA\SOFTENG 700A\P4P-Speech2Gesture\save\speech2gesture\oliver\exp_105_cpk_speech2gesture_speaker_['oliver']_model_Speech2Gesture_G_note_speech2gesture_weights.p" -render_text 0 -path2data "F:\PATSDATASET\oliver\pats\data"
```

## Codebase introduction

### gan.py

Focuses on the GAN framework, including the training loop, loss calculations, and integration of generator and discriminator for the training process

### speech2gesture.py

Defines the specific architectures of the generator and discriminator models used within the GAN framework

## Dependencies

#### scipy: `version 1.10.1`

#### gensim: TBD

#### webrtcvad: TBD (install visual studio build tool on Windows)

## Bug report

### style_classifier.py - line 20

Change default num of speaker from input to 25 if pretrained model (trainer.py - line 405 & 406) is activated

### dataUtils.py - line 138

Update pandas data structure operation

### dataUtils.py - line 277

Convert byte strings in missing_intervals to regular strings

### trainer.py - line 62

have to manually replace speaker arugment if pretrained model (line 405 & 406) is activated otherwise it would become shelly

### trainer.py - line 405 & 406

No pretrained model

### argUtils.py - Dataset Parameters

Updated some parameters to ensure compatibility issue with Windows, dataset storage, data type

### speech2gesture.py - line 30

Added missing field kwargs

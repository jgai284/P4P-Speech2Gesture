# P4P-Speech2Gesture

## Class introduction

### gan.py

Focuses on the GAN framework, including the training loop, loss calculations, and integration of generator and discriminator for the training process

### speech2gesture.py

Defines the specific architectures of the generator and discriminator models used within the GAN framework

## Dependencies

#### scipy: version 1.10.1

#### gensim: TBD

#### webrtcvad: TBD (install visual studio build tool on Windows)

## Bug report

### style_classifier.py - line 20
Change default num of speaker from 1 to 25

### dataUtils.py - line 138
Update pandas data structure operation

### dataUtils.py - line 277
Convert byte strings in missing_intervals to regular strings

### trainer.py - line 62
have to manually replace speaker arugment otherwise it would become shelly

### argUtils.py - Dataset Parameters
Updated some parameters to ensure compatibility issue with Windows, dataset storage, data type

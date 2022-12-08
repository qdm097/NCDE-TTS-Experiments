# NCDE-TTS-Experiments

This repository was created in association with a course (Comp 562: Machine Learning) for the purpose of writing a paper (research started in January 2020, but repo released now for said purpose).
It is the result of about 2 years of on and off experimentation and research into NCDEs in the context of speech synthesizers.
The paper associated with that class may be added to this repo at some point to help with understanding, but in brief:\
\
This project is a modified version of [Tacotron2](https://github.com/NVIDIA/tacotron2). In order to use it one must first install that project and then replace the files with the same name as in this repository.\
One must then install [Neural CDEs](https://github.com/patrick-kidger/NeuralCDE) and replace the file "ncde.py" from src/ncde/ncde.py with the one hosted here. This is because an updated solver was released that the original repositiory did not take into account (reversible heun) which has increased performance in this context. It can also be used as an adjoint method.\
Any other imports must then be resolved, but these are the main two.\
\
Note that due to thinking this repository would not be released for reasons to be detailed soon the project is a bit of a mess.
Most importantly is that the NCDE repo is not imported in the traditional way. This is because at least one file was modified in it. Instead it is a source root in my IDE (PyCharm) with a sys.path.append("x") call to the root folder.\
\
As for why this project was not planned to be released: its stated goal failed. NCDEs for TTS are promising, but due to issues with linking encoded text and teacher-forced mel speech samples the project was unable to achieve lower loss than Tacotron2 when properly masking the transformer decoder used for that process. Without masking, however, the project was able to achieve loss of 0.5 (same as TT2) by 23,000 iterations (at roughly 30s/it on an RTX Strix 1080). Also, neither model is capable of inference. The masked model can generate single words from a full sentence used as text input, but always gets stuck in a feedback loop. As such the project is deemed a failure, even if it has been enlightening. Note, though, that output speech has a very natural tone, despite the fact it's only single words or a few syllables. I speculate this is the result of NCDEs being highly suited for this but transformers being limited in the context. It could also be a parameter or implementation issue, but point being that the project is done for now. I may work on NCDEs in the context of singing synthesizers (since no transformer would be necessary there), but that's tbd.

# EC523_Project

## Introduction
This repo contains BirdNET and Google Sound Seperation models and scripts for processing large amounts of audio data or single audio files. This repo contains both the latest and older versions of BirdNET for acoustic analyses.

## Setup (birdnetlib)

The easiest way to setup BirdNET on your machine is to install [birdnetlib](https://pypi.org/project/birdnetlib/) through pip with:

```
pip3 install birdnetlib
```

Make sure to install Tensorflow Lite, librosa and ffmpeg like mentioned below. You can run BirdNET with:

```
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from datetime import datetime

# Load and initialize the BirdNET-Analyzer models.
analyzer = Analyzer()

recording = Recording(
    analyzer,
    "sample.mp3",
    lat=35.4244,
    lon=-120.7463,
    date=datetime(year=2022, month=5, day=10), # use date or week_48
    min_conf=0.25,
)
recording.analyze()
print(recording.detections)
```


## References

[1]Stefan Kahl, et al. "Birdnet: A Deep Learning Solution for Avian Diversity Monitoring." Ecological informatics, v.61. pp. 101236. 

[2]Schlüter, Jan. "Learning to Monitor Birdcalls From Weakly-Labeled Focused Recordings." CLEF (Working Notes). 2021.

[3]Xie, Jiangjian, et al. "A review of automatic recognition technology for bird vocalizations in the deep learning era." Ecological Informatics (2022): 101927.

[4]LeBien, Jack et al. “A pipeline for identification of bird and frog species in tropical soundscape recordings using a convolutional neural network.” Ecol. Informatics 59 (2020): 101113.

[5]G'omez-G'omez, Juan et al. “Western Mediterranean wetlands bird species classification: evaluating small-footprint deep learning approaches on a new annotated dataset.” ArXiv abs/2207.05393 (2022): n. pag.

[6]Scott Wisdom, Efthymios Tzinis, Hakan Erdogan, Ron J. Weiss, Kevin Wilson, John R. Hershey, "Unsupervised Sound Separation Using Mixture Invariant Training", Advances in Neural Information Processing Systems, 2020.

[7]Tom Denton, Scott Wisdom, John R. Hershey, "Improving Bird Classification with Unsupervised Sound Separation", Proc. IEEE International Conference on Audio, Speech, and Signal Processing (ICASSP), 2022.

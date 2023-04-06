# EC523_Project

## BirdNet

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

## Usage

1. Inspect config file for options and settings, especially inference settings. Specify a custom species list if needed and adjust the number of threads TFLite can use to run the inference.

2. Run `analyzer.py` to analyze an audio file. You need to set paths for the audio file and selection table output. Here is an example:

```
python3 analyze.py --i /path/to/audio/folder --o /path/to/output/folder
```

<b>NOTE</b>: Your custom species list has to be named 'species_list.txt' and the folder containing the list needs to be specified with `--slist /path/to/folder`. You can also specify the number of CPU threads that should be used for the analysis with `--threads <Integer>` (e.g., `--threads 16`). If you provide GPS coordinates with `--lat` and `--lon`, the custom species list argument will be ignored.

3. Run `embeddings.py` to extract feature embeddings instead of class predictions. Result file will contain timestamps and lists of float values representing the embedding for a particular 3-second segment. Embeddings can be used for clustering or similarity analysis. Here is an example:

```
python3 embeddings.py --i example/ --o example/ --threads 4 --batchsize 16
```

4. After the analysis, run `segments.py` to extract short audio segments for species detections to verify results. This way, it might be easier to review results instead of loading hundreds of result files manually.

5. When editing your own `species_list.txt` file, make sure to copy species names from the labels file of each model. 

6. You can generate a species list for a given location using `species.py` in case you need it for reference. Here is an example:

```
python3 species.py --o example/species_list.txt --lat 42.5 --lon -76.45 --week 4
```


## Google Sound Seperation

## Model checkpoints

Two model checkpoints, one with 4 output sources and one with 8 output sources, are available on Google Cloud. These models assume input audio sampled at 22.05 kHz. The models can be downloaded using the following command, which will copy the model checkpoint files to the current folder:

```
gsutil -m cp -r gs://gresearch/sound_separation/bird_mixit_model_checkpoints .
```


## Install TensorFlow
Follow the instructions
<a href="https://www.tensorflow.org/install">here</a>.


## Run the model on a wav file.

Once you have installed TensorFlow, you can run the 4-output model on a wav file using the following:

```
python3 ../tools/process_wav.py \
--model_dir bird_mixit_model_checkpoints/output_sources4 \
--checkpoint bird_mixit_model_checkpoints/output_sources4/model.ckpt-3223090 \
--num_sources 4 \
--input <input name>.wav \
--output <output_name>.wav
```
which will result in 4 wav files `<output_name>_source0.wav`, ... , `<output_name>_source3.wav`.

The 8-output model can be run using the following:

```
python3 ../tools/process_wav.py \
--model_dir bird_mixit_model_checkpoints/output_sources8 \
--checkpoint bird_mixit_model_checkpoints/output_sources8/model.ckpt-2178900 \
--num_sources 8 \
--input <input name>.wav \
--output <output_name>.wav
```
which will result in 8 wav files `<output_name>_source0.wav`, ... , `<output_name>_source7.wav`.



## References

[1]Stefan Kahl, et al. "Birdnet: A Deep Learning Solution for Avian Diversity Monitoring." Ecological informatics, v.61. pp. 101236. 

[2]Schlüter, Jan. "Learning to Monitor Birdcalls From Weakly-Labeled Focused Recordings." CLEF (Working Notes). 2021.

[3]Xie, Jiangjian, et al. "A review of automatic recognition technology for bird vocalizations in the deep learning era." Ecological Informatics (2022): 101927.

[4]LeBien, Jack et al. “A pipeline for identification of bird and frog species in tropical soundscape recordings using a convolutional neural network.” Ecol. Informatics 59 (2020): 101113.

[5]G'omez-G'omez, Juan et al. “Western Mediterranean wetlands bird species classification: evaluating small-footprint deep learning approaches on a new annotated dataset.” ArXiv abs/2207.05393 (2022): n. pag.

[6]Scott Wisdom, Efthymios Tzinis, Hakan Erdogan, Ron J. Weiss, Kevin Wilson, John R. Hershey, "Unsupervised Sound Separation Using Mixture Invariant Training", Advances in Neural Information Processing Systems, 2020.

[7]Tom Denton, Scott Wisdom, John R. Hershey, "Improving Bird Classification with Unsupervised Sound Separation", Proc. IEEE International Conference on Audio, Speech, and Signal Processing (ICASSP), 2022.

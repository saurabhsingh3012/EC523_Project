# 80524CommonNighthawkCardenAlvar.wav
# test1_source1.wav


import tensorflow as tf
import metrics
from metrics import signal_to_noise_ratio_gain_invariant

# Read the audio file
original_binary = tf.io.read_file('80524CommonNighthawkCardenAlvar.wav')
separated_binary = tf.io.read_file('test1_source0.wav')

# Decode the audio file to a tensor
original_audio, original_sample_rate = tf.audio.decode_wav(original_binary)
separated_audio, separated_sample_rate = tf.audio.decode_wav(separated_binary)

# Print the shape and sample rate of the audio tensor
print('Original tensfor:',original_audio[1])
print('Audio shape:', original_audio.shape)
print('Sample rate:', original_sample_rate)

# Print the shape and sample rate of the audio tensor
print('Separated tensor:',separated_audio)
print('Audio shape:', separated_audio.shape)
print('Sample rate:', separated_sample_rate)

import tensorflow as tf
import tensorflow_io as tfio

audio = tfio.audio.AudioIOTensor('80524CommonNighthawkCardenAlvar.wav')

print('aaaudio:',audio)


# Calculate the SI-SNR metric between the original and separated audio.
#si_snr = metrics.calculate_signal_to_noise_ratio(separated_audio,original_audio)

# Print the SI-SNR value.
#print('SI-SNR:', si_snr)


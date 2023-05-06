# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Process a .wav file with a separation model.

Example usage:

python3 ../tools/process_wav_Saurabh.py \
--model_dir ../bird_mixit/output_sources4 \
--checkpoint ../bird_mixit/output_sources4/model.ckpt-3223090 \
--num_sources 4 \
--input_dir ../../../../data \
--output_dir ../../../../xc_ny_soundscape_seperated
"""
# pylint: enable=line-too-long

import argparse
import os
import inference

import tensorflow.compat.v1 as tf

strtobool = inference.strtobool


def main():
  parser = argparse.ArgumentParser(
      description='Process a mixture .wav file to separate into sources.')
  parser.add_argument(
      '-i', '--input_dir', help='Input directory containing .wav files.', required=True, type=str)
  parser.add_argument(
      '-o', '--output_dir', help='Output directory to save .wav files.', required=True, type=str)
  parser.add_argument(
      '-m', '--model_dir', help='Model root directory, required. '
      'Must contain inference.meta and at least one checkpoint.', type=str)
  parser.add_argument(
      '-ic', '--input_channels', help='Truncate/pad input to this many '
      'channels, if positive.', default=0, type=int)
  parser.add_argument(
      '-ns', '--num_sources', help='Number of output sources in the model.',
      default=2, type=int)
  parser.add_argument(
      '-oc', '--output_channels', help='Truncate output to this many channels, '
      'i.e. sources, if positive.', default=0, type=int)
  parser.add_argument(
      '-it', '--input_tensor', default='input_audio/receiver_audio:0',
      help='Name of tensor to which to feed input_wav.', type=str)
  parser.add_argument(
      '-ot', '--output_tensor', default='denoised_waveforms:0',
      help='Name of tensor to output as output_wav.', type=str)
  parser.add_argument(
      '-si', '--scale_input', default=False, help='If True, scale the input '
      'signal such that its absolute maximum value is 0.99.', type=strtobool)
  parser.add_argument(
      '-c', '--checkpoint', default=None, help='Override for checkpoint path.')
  parser.add_argument(
      '-wos', '--write_outputs_separately', default=True,
      help='Write output source signals into separate wav files.',
      type=strtobool)
  args = parser.parse_args()

  tf.disable_v2_behavior()

  meta_graph_filename = os.path.join(args.model_dir, 'inference.meta')
  tf.logging.info('Importing meta graph: %s', meta_graph_filename)

  with tf.Graph().as_default() as g:
    saver = tf.train.import_meta_graph(meta_graph_filename)
    meta_graph_def = g.as_graph_def()

  tf.reset_default_graph()

  input_dir = args.input_dir
  output_dir = args.output_dir
  if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    
  processed_file_path = "../../../../xc_ny_soundscape_seperated/processed_files.txt"

  # Load the list of processed files, or create an empty list if the file doesn't exist yet.
  if os.path.isfile(processed_file_path):
      with open(processed_file_path, "r") as f:
          processed_files = [line.strip() for line in f]
  else:
      processed_files = []


  for file_name in os.listdir(input_dir):
      # Check if the file has already been processed.
      if file_name in processed_files:
          print(f"Skipping {file_name} as it has already been processed.")
          continue
      
      try:
            if file_name.endswith(".wav"):
                tf.reset_default_graph()#######################
                input_wav, sample_rate = inference.read_wav_file(
                    os.path.join(input_dir, file_name), args.input_channels, args.scale_input)
                print(file_name)
  
                input_wav = tf.transpose(input_wav)
                input_wav = tf.expand_dims(input_wav, axis=0)  # shape: [1, mics, samples]
                output_wav, = tf.import_graph_def(
                    meta_graph_def,
                    name='',
                    input_map={args.input_tensor: input_wav},
                    return_elements=[args.output_tensor])
  
                output_wav = tf.squeeze(output_wav, 0)  # shape: [sources, samples]
                output_wav = tf.transpose(output_wav)
                if args.output_channels > 0:
                    output_wav = output_wav[:, :args.output_channels]
  
                #if not os.path.exists(os.path.join(output_dir, file_name)):
                  #os.makedirs(os.path.join(output_dir, file_name))
  
                output = os.path.join(output_dir, file_name[:-4])
                output = os.path.join(output, file_name)
                #print(output)
  
                write_output_ops = inference.write_wav_file(
                    output, output_wav, sample_rate=sample_rate,
                    num_channels=args.num_sources,
                    output_channels=args.output_channels,
                    write_outputs_separately=args.write_outputs_separately,
                    channel_name='source')
                #print("1************************************")
  
                checkpoint = args.checkpoint
                if not checkpoint:
                  checkpoint = tf.train.latest_checkpoint(args.model_dir)
                  #print("2************************************")
                with tf.Session() as sess:
                  sess.run(tf.global_variables_initializer())#################################
                  tf.logging.info('Restoring from checkpoint: %s', checkpoint)
                  saver.restore(sess, checkpoint)
                  sess.run(write_output_ops)
                #print("3************************************")
              
      except Exception as e:
              print(f"Error processing {file_name}: {e}")
              continue
        
      # Mark the file as processed by adding it to the list of processed files.
      processed_files.append(file_name)
      with open(processed_file_path, "a") as f:
          f.write(f"{file_name}\n")


if __name__ == '__main__':
  main()


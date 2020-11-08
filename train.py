# Copyright 2018 Google LLC
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
"""A demo script showing how to use the uisrnn package on toy data."""

import numpy as np

import uisrnn


SAVED_MODEL_NAME = 'saved_model.uisrnn'


def diarization_train(model_args, training_args, inference_args):
  """Experiment pipeline.

  Load data --> train model --> test model --> output result

  Args:
    model_args: model configurations
    training_args: training configurations
    inference_args: inference configurations
  """

  predicted_cluster_ids = []
  test_record = []

  train_data = np.load('./data/toy_training_data.npz', allow_pickle=True)
  test_data = np.load('./data/toy_testing_data.npz', allow_pickle=True)
  train_sequence = train_data['train_sequence'] #shape (47350, 256)
  train_cluster_id = train_data['train_cluster_id'] # shape (47350,)
  test_sequences = test_data['test_sequences'].tolist() # a list of 25 elements, each element has shape (seq_len, 256)
  test_cluster_ids = test_data['test_cluster_ids'].tolist()

  model = uisrnn.UISRNN(model_args)

  # Training.
  # If we have saved a mode previously, we can also skip training by
  # calling：
  # model.load(SAVED_MODEL_NAME)
  model.fit(train_sequence, train_cluster_id, training_args)
  model.save(SAVED_MODEL_NAME)

def main():
  """The main function."""
  model_args, training_args, inference_args = uisrnn.parse_arguments()
  diarization_train(model_args, training_args, inference_args)


if __name__ == '__main__':
  main()

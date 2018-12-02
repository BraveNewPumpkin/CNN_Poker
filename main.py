import sys
import gc
import time
import pickle
from pathlib import Path
from numbers import Number
from collections import Set, Mapping, deque
from klepto.archives import dir_archive
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
import numpy as np
from sklearn.model_selection import train_test_split

import dealer
import self_play

def main(args):
  # reward_dict = dealer.run(50)
  #
  # gc.collect()
  #
  # save_obj(reward_dict, 'heuristic')

  # reward_dict = {}
  # with open("training_data/heuristic.pkl", "rb") as f:
  #     reward_dict = pickle.load(f)

  reward_dict = load_obj('heuristic')

  model = train(reward_dict)

  del reward_dict
  gc.collect()

  models_dirpath = Path('models')
  model.save(str(models_dirpath / "heuristic.model"))

  for i in range(1, 8):
    reward_dict = self_play.run(100000, model)

    gc.collect()

    save_obj(reward_dict, 'self_play_' + str(i))
    model = train(reward_dict)

    del reward_dict
    gc.collect()

    self_play_name = "self_play_" + str(i) + ".model"
    model.save(str(models_dirpath / self_play_name))

  return 0

def train(reward_dict):
  input_shape = [9, 17, 17]
  output_shape = [3]

  test_size = 10000
  x_train, x_test, y_train, y_test = extract_train_and_test(reward_dict, input_shape, test_size)

  steps_per_epoch = int(int(x_train.shape[0]) / 512)
  # steps_per_epoch / num_validation_steps == num_training_examples / num_testing_examples
  num_classes = 3
  epochs = 6

  tensor_board_path = Path('logs/{}'.format(time.time()))
  tensor_board = TensorBoard(log_dir=str(tensor_board_path),
                             write_graph=True,
                             write_images=True,
                             )

  model = create_model(input_shape, num_classes)

  # tensor_board.set_model(model)

  model.fit(x_train,
            y_train,
            steps_per_epoch=steps_per_epoch,
            validation_steps=test_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test),
            # callbacks=[tensor_board])
            )
  score = model.evaluate(x_test, y_test, verbose=0)
  print('Test loss:', score[0])
  print('Test accuracy:', score[1])

  return model


def extract_train_and_test(reward_dict, input_shape, test_size):
  x_all_raw, y_all_raw = extract_x_and_y(reward_dict)

  x_train, x_test, y_train, y_test = train_test_split(x_all_raw, y_all_raw, train_size=25000, test_size=test_size)
  x_train = pad_input(input_shape, x_train)
  x_test = pad_input(input_shape, x_test)

  return x_train, x_test, y_train, y_test

def extract_x_and_y(reward_dict):
  x_all_serialized = reward_dict.keys()
  x_all_raw = np.array([pickle.loads(k) for k in x_all_serialized])
  y_all_raw = np.array(list(reward_dict.values()))
  return x_all_raw, y_all_raw


def pad_input(desired_input_shape, x):
  right_left_pad = desired_input_shape[1] - x.shape[2]
  left_pad = right_left_pad // 2
  right_pad = left_pad + (right_left_pad % 2)
  top_bottom_pad = desired_input_shape[2] - x.shape[3]
  top_pad = top_bottom_pad // 2
  bottom_pad = top_pad + (top_bottom_pad % 2)

  tf_padding_dims = [[0, 0], [0, 0], [left_pad, right_pad], [top_pad, bottom_pad]]

  x_padded = np.pad(x,  ((0, 0), (0, 0), (left_pad, right_pad), (top_pad, bottom_pad)), mode='constant')
  # x_padded = tf.pad(x, tf_padding_dims)
  return x_padded

def create_model(input_shape, num_classes):
  model = Sequential()
  initial = Conv2D(32,
                   kernel_size=(3, 3),
                   padding='same',
                   activation='relu',
                   input_shape=input_shape,
                   data_format='channels_first'
                   )
  model.add(initial)
  model.add(Conv2D(64,
                   kernel_size=(3, 3),
                   activation='relu'
                   ))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Conv2D(128,
                   kernel_size=(3, 3),
                   activation='relu',
                   ))
  model.add(Conv2D(128,
                   kernel_size=(3, 3),
                   activation='relu'
                   ))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Flatten())
  model.add(Dropout(0.5))
  model.add(Dense(num_classes, activation='linear'))

  model.compile(
    loss=keras.losses.mean_squared_error,
    optimizer=keras.optimizers.Nadam(lr=0.1),
    metrics=['accuracy']
  )

  return model


def save_obj(dict, name):
  training_data_path_str = str(Path('training_data') / name)
  heuristic_training_dict = dir_archive(name=training_data_path_str, dict=dict, cached=False)
  heuristic_training_dict.dump()

def load_obj(name):
  training_data_path_str = str(Path('training_data') / name)
  heuristic_training_dict = dir_archive(name=training_data_path_str)
  heuristic_training_dict.load()
  return heuristic_training_dict


zero_depth_bases = (str, bytes, Number, range, bytearray)
iteritems = 'items'

def getsize(obj_0):
    """Recursively iterate to sum size of object & members."""
    _seen_ids = set()
    def inner(obj):
        obj_id = id(obj)
        if obj_id in _seen_ids:
            return 0
        _seen_ids.add(obj_id)
        size = sys.getsizeof(obj)
        if isinstance(obj, zero_depth_bases):
            pass # bypass remaining control flow and return
        elif isinstance(obj, (tuple, list, Set, deque)):
            size += sum(inner(i) for i in obj)
        elif isinstance(obj, Mapping) or hasattr(obj, iteritems):
            size += sum(inner(k) + inner(v) for k, v in getattr(obj, iteritems)())
        # Check for custom object instances - may subclass above too
        if hasattr(obj, '__dict__'):
            size += inner(vars(obj))
        if hasattr(obj, '__slots__'): # can have __slots__ with __dict__
            size += sum(inner(getattr(obj, s)) for s in obj.__slots__ if hasattr(obj, s))
        return size
    return inner(obj_0)



main(sys.argv)

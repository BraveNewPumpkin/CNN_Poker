import sys
import time
import pickle
import keras
from pathlib import Path
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
import numpy as np
from sklearn.model_selection import train_test_split

import dealer
import self_play


def main(args):
  reward_dict = dealer.run(5)

  save_obj(reward_dict, 'heuristic')

  model = train(reward_dict)

  models_dirpath = Path('models')
  model.save(str(models_dirpath / "heuristic.model"))

  for i in range(1, 3):
    reward_dict = self_play.run(5, model)
    save_obj(reward_dict, 'self_play_' + str(i))
    model = train(reward_dict)

    self_play_name = "self_play_" + str(i) + ".model"
    model.save(str(models_dirpath / self_play_name))

  return 0

def train(reward_dict):
  input_shape = [9, 17, 17]
  output_shape = [3]

  x_train, x_test, y_train, y_test = extract_train_and_test(reward_dict, input_shape)

  steps_per_epoch = 10
  # steps_per_epoch / num_validation_steps == num_training_examples / num_testing_examples
  num_classes = 3
  epochs = 6

  tensor_board_path = Path('logs/{}'.format(time.time()))
  tensor_board = TensorBoard(log_dir=str(tensor_board_path),
                             write_graph=True,
                             write_images=True,
                             )

  model = create_model(input_shape, num_classes)

  tensor_board.set_model(model)

  model.fit(x_train,
            y_train,
            steps_per_epoch=steps_per_epoch,
            validation_steps=21,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test),
            callbacks=[tensor_board])
  score = model.evaluate(x_test, y_test, verbose=0)
  print('Test loss:', score[0])
  print('Test accuracy:', score[1])

  return model


def extract_train_and_test(reward_dict, input_shape):
  x_all_raw, y_all_raw = extract_x_and_y(reward_dict)
  # print('x_all_raw: ', x_all_raw.shape)
  x_all_padded = pad_input(input_shape, x_all_raw)
  # print('x_all_raw padded: ', x_all_raw.shape)

  x_train, x_test, y_train, y_test = train_test_split(x_all_padded, y_all_raw, test_size=0.25)
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
  x_padded = np.pad(x,  ((0, 0), (0, 0), (left_pad, right_pad), (top_pad, bottom_pad)), mode='constant')
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
    optimizer=keras.optimizers.Nadam(lr=0.02),
    metrics=['accuracy']
  )

  return model

def save_obj(obj, name):
  with open('training_data/' + name + '.pkl', 'wb') as f:
    pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

main(sys.argv)

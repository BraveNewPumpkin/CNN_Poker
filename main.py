import sys
import pickle
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

import dealer


def main(args):
  input_shape = [9, 17, 17]
  output_shape = [3]

  reward_dict = dealer.run(256)
  x_all_serialized = reward_dict.keys()
  x_all_raw = np.array([pickle.loads(k) for k in x_all_serialized])
  print('x_all_raw: ', x_all_raw.shape)
  right_left_pad = input_shape[1] - x_all_raw.shape[2]
  left_pad = right_left_pad // 2
  right_pad = left_pad + (right_left_pad % 2)
  top_bottom_pad = input_shape[2] - x_all_raw.shape[3]
  top_pad = top_bottom_pad // 2
  bottom_pad = top_pad + (top_bottom_pad % 2)
  x_all_raw = np.pad(x_all_raw,  ((0, 0), (0, 0), (left_pad, right_pad), (top_pad, bottom_pad)), mode='constant')
  print('x_all_raw padded: ', x_all_raw.shape)
  y_all_raw = np.array(list(reward_dict.values()))

  x_train_raw, x_test_raw, y_train_raw, y_test_raw = train_test_split(x_all_raw, y_all_raw, test_size=0.25)

  num_training_examples = len(x_all_serialized)
  num_testing_examples = len(y_all_raw)
  num_validation_steps = 21
  steps_per_epoch = 10
  # steps_per_epoch / num_validation_steps == num_training_examples / num_testing_examples
  num_classes = 3
  epochs = 6

  print('num_training_examples: ', num_training_examples)
  print('xtrain_raw: ', x_train_raw.shape)


  x_train = x_train_raw
  x_test = x_test_raw
  y_train = y_train_raw
  y_test = y_test_raw

  print('x_train: ', x_train.shape)

  TensorBoard(log_dir='C:\cygwin64\home\lates\dev\cNN\project\logs\\')

  model = Sequential()
  initial = Conv2D(32,
                  kernel_size=(3, 3),
                  padding='same',
                  activation='relu',
                  input_shape=input_shape,
                  data_format='channels_first'
                  )
  conv32 = Conv2D(32,
                  kernel_size=(3, 3),
                  activation='relu',
                  )
  conv64 = Conv2D(64,
                   kernel_size=(3, 3),
                   activation='relu'
                   )
  model.add(initial)
  model.add(conv64)
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Conv2D(32,
                  kernel_size=(3, 3),
                  activation='relu',
                  ))
  model.add(Conv2D(64,
                   kernel_size=(3, 3),
                   activation='relu'
                   ))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Flatten())
  model.add(Dropout(0.5))
  model.add(Dense(num_classes, activation='linear'))

  model.compile(
    loss=keras.losses.mean_squared_error,
    optimizer=keras.optimizers.Adam(lr=0.02),
    metrics=['accuracy'])

  model.fit(x_train,
            y_train,
            steps_per_epoch=steps_per_epoch,
            validation_steps=21,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))
  score = model.evaluate(x_test, y_test, verbose=0)
  print('Test loss:', score[0])
  print('Test accuracy:', score[1])

  return 0


main(sys.argv)

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

  reward_dict = dealer.run(256)
  x_all_serialized = reward_dict.keys()
  x_all_raw = [pickle.loads(k) for k in x_all_serialized]
  y_all_raw = list(reward_dict.values())

  x_train_raw, x_test_raw, y_train_raw, y_test_raw = train_test_split(x_all_raw, y_all_raw, test_size=0.25)

  num_training_examples = len(x_all_serialized)
  num_testing_examples = len(y_all_raw)
  num_validation_steps = 21
  steps_per_epoch = 43
  # steps_per_epoch / num_validation_steps == num_training_examples / num_testing_examples
  batch_size = 256
  num_classes = 3
  epochs = 6
  # input_shape = [batch_size, 4, 13, 9]
  input_shape = [batch_size, 17, 17, 9]
  output_shape = [batch_size, 3]


  x_train = tf.placeholder(tf.float32, shape=input_shape, name='inputs')
  x_test = tf.placeholder(tf.float32, shape=input_shape, name='inputs')
  y_train = tf.placeholder(tf.float32, shape=output_shape, name='outputs')
  y_test = tf.placeholder(tf.float32, shape=output_shape, name='outputs')

  #TODO pad input to make it 17x17x9

  TensorBoard(log_dir='C:\cygwin64\home\lates\dev\cNN\project\logs\\')

  model = Sequential()
  layer1 = Conv2D(32,
                  kernel_size=(3, 3),
                  activation='relu',
                  input_shape=[17, 17, 9],
                  data_format='channels_last'
                  )
  # )
  model.add(layer1)
  model.add(Conv2D(64,
                   kernel_size=(3, 3),
                   activation='relu'
                   ))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))
  model.add(Flatten())
  model.add(Dense(128, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(num_classes, activation='linear'))

  model.compile(
    loss=keras.losses.mean_squared_error,
    optimizer=keras.optimizers.Adam(),
    metrics=['accuracy'])

  model.fit(x_train, y_train,
            steps_per_epoch=43,
            validation_steps=21,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))
  score = model.evaluate(x_test, y_test, verbose=0)
  print('Test loss:', score[0])
  print('Test accuracy:', score[1])

  return 0


main(sys.argv)

import sys
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf
import numpy as np


def main(args):
  batch_size = 128
  num_classes = 10
  epochs = 6
  input_shape = [1, 255, 255, 3]
  output_shape = [1, 255, 255, 3]

  x_train = tf.placeholder(tf.float32, shape=input_shape, name='inputs')
  x_test = tf.placeholder(tf.float32, shape=input_shape, name='inputs')
  y_train = tf.placeholder(tf.float32, shape=output_shape, name='outputs')
  y_test = tf.placeholder(tf.float32, shape=output_shape, name='outputs')

  keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=0, batch_size=batch_size, write_graph=True, write_grads=False,
                              write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                              embeddings_metadata=None, embeddings_data=None)

  model = Sequential()
  model.add(Conv2D(1, kernel_size=(3, 3),
                   activation='relu',
                   input_shape=[3, 255, 1]))
                   # activation='relu'))
  model.add(Conv2D(64, kernel_size=(3, 3),
                   activation='relu',
                   input_shape=[1, 253, 1]))
                   # activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))
  model.add(Flatten())
  model.add(Dense(128, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(num_classes, activation='linear'))

  model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])

  # model.fit(x_train, y_train,
  #           batch_size=batch_size,
  #           epochs=epochs,
  #           verbose=1,
  #           validation_data=(x_test, y_test))
  # score = model.evaluate(x_test, y_test, verbose=0)
  # print('Test loss:', score[0])
  # print('Test accuracy:', score[1])

  return 0


main(sys.argv)

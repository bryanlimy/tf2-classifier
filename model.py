import tensorflow as tf
import tensorflow.keras as keras


def get_model(hparams):
  if hparams.model == 'sequential':
    return sequential(hparams)
  else:
    return Model(hparams)


def sequential(hparams):
  model = keras.models.Sequential([
      keras.layers.Flatten(),
      keras.layers.Dense(128, activation='relu'),
      keras.layers.Dense(64, activation='relu'),
      keras.layers.Dense(hparams.num_classes, activation='softmax')
  ])
  return model


class Dense(keras.layers.Layer):

  def __init__(self, units, activation=None, name="dense"):
    super().__init__(name)
    self._units = units
    self._activation = activation

  def build(self, input_shape):
    self.weight = self.add_variable(
        'weight', shape=[int(input_shape[-1]), self._units], dtype=tf.float32)
    self.bias = self.add_variable('bias', shape=[self._units], dtype=tf.float32)

  def call(self, x):
    y = tf.matmul(x, self.weight) + self.bias
    if self._activation is not None:
      y = self._activation(y)
    return y


class Model(keras.Model):

  def __init__(self, hparams):
    super().__init__()
    self.flatten = keras.layers.Flatten()
    self.dense1 = Dense(units=128, activation=keras.activations.relu)
    self.dense2 = Dense(units=64, activation=keras.activations.relu)
    self.dense3 = Dense(
        units=hparams.num_classes, activation=keras.activations.softmax)

  def call(self, inputs):
    output = self.flatten(inputs)
    output = self.dense1(output)
    output = self.dense2(output)
    output = self.dense3(output)
    return output

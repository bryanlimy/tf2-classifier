import os
import time
import tensorflow as tf
import tensorflow.keras as keras
from utils import get_hparams, get_optimizer, Summary, get_dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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
    self.dense1 = Dense(
        units=hparams.num_units,
        activation=keras.activations.relu,
    )
    self.dense2 = Dense(
        units=hparams.num_units,
        activation=keras.activations.relu,
    )
    self.dense3 = Dense(
        units=hparams.num_classes,
        activation=keras.activations.softmax,
    )
    self.dropout = keras.layers.Dropout(rate=hparams.dropout)

  def call(self, inputs):
    output = self.flatten(inputs)
    output = self.dense1(output)
    output = self.dropout(output)
    output = self.dense2(output)
    output = self.dropout(output)
    output = self.dense3(output)
    return output


def loss_fn(labels, predictions):
  return keras.losses.SparseCategoricalCrossentropy()(labels, predictions)


@tf.function
def train_step(features, labels, model, optimizer):
  with tf.GradientTape() as tape:
    predictions = model(features)
    loss = loss_fn(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  return loss, predictions


@tf.function
def test_step(features, labels, model):
  predictions = model(features)
  loss = loss_fn(labels, predictions)
  return loss, predictions


def main():
  hparams = get_hparams()
  datasets = get_dataset(hparams)

  model = Model(hparams)

  optimizer = get_optimizer(hparams)

  summary = Summary(hparams, optimizer)

  for epoch in range(hparams.epochs):

    elapse = time.time()
    for images, labels in datasets['train']:
      summary.write_images(images, mode='train')
      loss, predictions = train_step(images, labels, model, optimizer)
      summary.log_progress(loss, labels, predictions, mode='train')

    summary.write_scalars(mode='train')
    elapse = time.time() - elapse

    for images, labels in datasets['test']:
      summary.write_images(images, mode='test')
      loss, predictions = test_step(images, labels, model)
      summary.log_progress(loss, labels, predictions, mode='test')

    summary.write_scalars(mode='test')
    summary.print_progress(epoch, elapse)


if __name__ == "__main__":
  main()

import os
import time
import tensorflow as tf
import tensorflow.keras as keras
from utils import get_hparams, get_dataset, get_optimizer, Logger

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


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
    self.dropout = keras.layers.Dropout(hparams.dropout)

  def call(self, inputs):
    output = self.flatten(inputs)
    output = self.dense1(output)
    output = self.dropout(output)
    output = self.dense2(output)
    output = self.dropout(output)
    output = self.dense3(output)
    return output


@tf.function
def train_step(features, labels, model, optimizer, loss_fn):
  with tf.GradientTape() as tape:
    predictions = model(features)
    loss = loss_fn(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  return loss, predictions


@tf.function
def test_step(features, labels, model, loss_fn):
  predictions = model(features)
  loss = loss_fn(labels, predictions)
  return loss, predictions


def train_and_test(hparams):
  datasets = get_dataset(hparams)
  model = Model(hparams)
  optimizer = get_optimizer(hparams)
  loss_fn = keras.losses.SparseCategoricalCrossentropy()
  logger = Logger(hparams, optimizer)

  for epoch in range(hparams.epochs):

    start = time.time()

    for images, labels in datasets['train']:
      loss, predictions = train_step(images, labels, model, optimizer, loss_fn)
      logger.log_progress(loss, labels, predictions, mode='train')

    elapse = time.time() - start

    logger.write_scalars(mode='train')

    for images, labels in datasets['test']:
      logger.write_images(images, mode='test')
      loss, predictions = test_step(images, labels, model, loss_fn)
      logger.log_progress(loss, labels, predictions, mode='test')

    logger.write_scalars(mode='test', elapse=elapse)
    logger.print_progress(epoch, elapse)


def main():
  hparams = get_hparams(output_root='runs/custom')
  train_and_test(hparams)


if __name__ == "__main__":
  main()

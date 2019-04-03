import os
import time
import tensorflow as tf
import tensorflow.keras as keras
from model import get_model
from utils import HParams, Logger, get_dataset, get_summary_writer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_hparams():
  hparams = HParams()
  hparams.epochs = 50
  hparams.batch_size = 32
  hparams.learning_rate = 0.001
  hparams.dataset = 'cifar10'
  hparams.model = 'sequential'
  hparams.loop = 'custom'
  hparams.output_dir = 'runs/002_cifar10_sequential_custom'
  return hparams


def simple(hparams, model, datasets):
  model.compile(
      optimizer=keras.optimizers.Adam(lr=hparams.learning_rate),
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy'])

  model.fit(
      x=datasets['train'],
      y=None,
      epochs=hparams.epochs,
      validation_data=datasets['test'])


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


def custom(hparams, model, datasets):
  logger = Logger(hparams)

  optimizer = keras.optimizers.Adam(lr=0.001)

  for epoch in range(hparams.epochs):

    elapse = time.time()

    for images, labels in datasets['train']:
      loss, predictions = train_step(images, labels, model, optimizer)
      logger.log_progress(loss, labels, predictions, mode='train')

    logger.write_summary(step=optimizer.iterations, mode='train')

    elapse = time.time() - elapse

    for images, labels in datasets['test']:
      loss, predictions = test_step(images, labels, model)
      logger.log_progress(loss, labels, predictions, mode='test')

    logger.write_summary(step=optimizer.iterations, mode='test')
    logger.print_progress(epoch, elapse)


def main():
  hparams = get_hparams()

  datasets = get_dataset(hparams)

  model = get_model(hparams)

  if hparams.loop == 'simple':
    simple(hparams, model, datasets)
  else:
    custom(hparams, model, datasets)

  model.summary()


if __name__ == "__main__":
  main()

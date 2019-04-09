import os
import time
import tensorflow as tf
import tensorflow.keras as keras
from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import summary as hparams_summary

from custom import Model
from utils import get_hparams, get_dataset, get_optimizer, Logger, \
  print_run_info, create_experiment_summary

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def train_step(features, labels, model, optimizer, loss_fn):
  with tf.GradientTape() as tape:
    predictions = model(features)
    loss = loss_fn(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  return loss, predictions


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

  summary_start = hparams_summary.session_start_pb(hparams=hparams.__dict__)

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

  summary_end = hparams_summary.session_end_pb(api_pb2.STATUS_SUCCESS)
  logger.write_hparams_summary(summary_start, summary_end, elapse)


def main():
  # list of hparams to test
  optimizer_list = ['adam', 'sgd']
  num_units_list = [32, 128]
  dropout_list = [0.0, 0.4, 0.8]

  exp_summary = create_experiment_summary(optimizer_list, num_units_list,
                                          dropout_list)
  root_writer = tf.summary.create_file_writer('runs/tuning')
  with root_writer.as_default():
    tf.summary.import_event(
        tf.compat.v1.Event(summary=exp_summary).SerializeToString())

  run = 0
  for optimizer in optimizer_list:
    for num_units in num_units_list:
      for dropout in dropout_list:
        hparams = get_hparams(
            num_units=num_units, optimizer=optimizer, dropout=dropout, run=run)
        print_run_info(hparams)
        train_and_test(hparams)
        run += 1


if __name__ == "__main__":
  main()

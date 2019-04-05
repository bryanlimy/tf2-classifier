import os
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_datasets as tfds

from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import summary as hparams_summary
from google.protobuf import struct_pb2


def get_hparams(num_units=128,
                optimizer='adam',
                learning_rate=0.001,
                dataset='fashion_mnist',
                run=0):
  hparams = HParams()
  hparams.epochs = 20
  hparams.batch_size = 64
  hparams.learning_rate = learning_rate
  hparams.num_units = num_units
  hparams.optimizer = optimizer
  hparams.dataset = dataset
  hparams.output_dir = 'runs/tuning/%03d_%s_lr%.4f_units%d' % (
      run, optimizer, learning_rate, num_units)
  return hparams


def get_dataset(hparams):
  dataset, info = tfds.load(
      name=hparams.dataset, with_info=True, as_supervised=True)

  hparams.num_classes = info.features['label'].num_classes

  def scale_image(image, label):
    return tf.cast(image, tf.float32) / 255.0, label

  dataset_train = dataset['train'].map(scale_image).shuffle(10000).batch(
      hparams.batch_size)
  dataset_test = dataset['test'].map(scale_image).batch(hparams.batch_size)

  return {'train': dataset_train, 'test': dataset_test}


def get_optimizer(hparams):
  if hparams.optimizer == 'adam':
    return keras.optimizers.Adam(lr=hparams.learning_rate)
  elif hparams.optimizer == 'sgd':
    return keras.optimizers.SGD(lr=hparams.learning_rate)
  elif hparams.optimizer == 'rmsprop':
    return keras.optimizers.RMSprop(lr=hparams.learning_rate)
  else:
    print('Optimizer %s not found' % hparams.optimizer)
    exit()


class HParams(object):
  """Empty object hyper-parameters """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)


class Logger(object):

  def __init__(self, hparams, optimizer):
    self.train_loss = keras.metrics.Mean(name="train_loss")
    self.train_accuracy = keras.metrics.SparseCategoricalAccuracy(
        name="train_accuracy")

    self.test_loss = keras.metrics.Mean(name="test_loss")
    self.test_accuracy = keras.metrics.SparseCategoricalAccuracy(
        name="test_accuracy")

    self.train_summary = tf.summary.create_file_writer(hparams.output_dir)
    self.test_summary = tf.summary.create_file_writer(
        os.path.join(hparams.output_dir, 'eval'))

    self.optimizer = optimizer

  def _step(self):
    return self.optimizer.iterations

  def log_progress(self, loss, labels, predictions, mode):
    if mode == 'train':
      self.train_loss(loss)
      self.train_accuracy(labels, predictions)
    else:
      self.test_loss(loss)
      self.test_accuracy(labels, predictions)

  def write_images(self, images, mode):
    summary = self.train_summary if mode == 'train' else self.test_summary
    # log 3 images
    with summary.as_default():
      tf.summary.image('features', images, step=self._step(), max_outputs=3)

  def write_scalars(self, mode, elapse=None):
    if mode == 'train':
      loss_metric = self.train_loss
      accuracy_metric = self.train_accuracy
      summary = self.train_summary
    else:
      loss_metric = self.test_loss
      accuracy_metric = self.test_accuracy
      summary = self.test_summary

    with summary.as_default():
      tf.summary.scalar('loss', loss_metric.result(), step=self._step())
      tf.summary.scalar('accuracy', accuracy_metric.result(), step=self._step())
      if elapse is not None:
        tf.summary.scalar(
            'elapse', elapse, step=self._step(), description='sec per epoch')

  def write_hparams_summary(self, summary_start, summary_end, elapse):
    with self.test_summary.as_default():
      # tf.summary.scalar('loss', self.test_loss.result(), step=self._step())
      # tf.summary.scalar(
      #     'accuracy', self.test_accuracy.result(), step=self._step())
      # tf.summary.scalar('elapse', elapse, step=self._step())

      tf.summary.import_event(
          tf.compat.v1.Event(summary=summary_start).SerializeToString())
      tf.summary.import_event(
          tf.compat.v1.Event(summary=summary_end).SerializeToString())

  def print_progress(self, epoch, elapse):
    template = 'Epoch {}, Loss {:.4f}, Accuracy: {:.2f}, Test Loss: {:.4f}, ' \
               'Test Accuracy: {:.2f}, Time: {:.2f}s'
    print(
        template.format(epoch, self.train_loss.result(),
                        self.train_accuracy.result() * 100,
                        self.test_loss.result(),
                        self.test_accuracy.result() * 100, elapse))


def create_experiment_summary(optimizer_list, num_units_list,
                              learning_rate_list):
  optimizer_list_val = struct_pb2.ListValue()
  optimizer_list_val.extend(optimizer_list)
  num_units_list_val = struct_pb2.ListValue()
  num_units_list_val.extend(num_units_list)
  learning_rate_list_val = struct_pb2.ListValue()
  learning_rate_list_val.extend(learning_rate_list)

  return hparams_summary.experiment_pb(
      hparam_infos=[
          api_pb2.HParamInfo(
              name='optimizer',
              display_name='Optimizer',
              type=api_pb2.DATA_TYPE_STRING,
              domain_discrete=optimizer_list_val),
          api_pb2.HParamInfo(
              name='num_units',
              display_name='Number of units',
              type=api_pb2.DATA_TYPE_FLOAT64,
              domain_discrete=num_units_list_val),
          api_pb2.HParamInfo(
              name='learning_rate',
              display_name='Learning rate',
              type=api_pb2.DATA_TYPE_FLOAT64,
              domain_discrete=learning_rate_list_val),
      ],
      metric_infos=[
          api_pb2.MetricInfo(
              name=api_pb2.MetricName(tag='accuracy'), display_name='Accuracy'),
          api_pb2.MetricInfo(
              name=api_pb2.MetricName(tag='elapse'), display_name='Elapse')
      ])


def print_run_info(hparams):
  print("\n-----------------------------------------\n"
        "BEGINNING RUN:\n"
        "\t num units: %d\n"
        "\t optimizer: %s\n"
        "\t learning rate: %.4f\n"
        "-----------------------------------------" % (
            hparams.num_units,
            hparams.optimizer,
            hparams.learning_rate,
        ))

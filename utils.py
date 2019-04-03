import os
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_datasets as tfds


class HParams(object):
  """Empty object hyper-parameters """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)


class Logger(object):

  def __init__(self, hparams):
    self.train_loss = keras.metrics.Mean(name="train_loss")
    self.train_accuracy = keras.metrics.SparseCategoricalAccuracy(
        name="train_accuracy")

    self.test_loss = keras.metrics.Mean(name="test_loss")
    self.test_accuracy = keras.metrics.SparseCategoricalAccuracy(
        name="test_accuracy")

    self.train_summary = tf.summary.create_file_writer(hparams.output_dir)
    self.test_summary = tf.summary.create_file_writer(
        os.path.join(hparams.output_dir, 'eval'))

  def log_progress(self, loss, labels, predictions, mode):
    if mode == 'train':
      self.train_loss(loss)
      self.train_accuracy(labels, predictions)
    else:
      self.test_loss(loss)
      self.test_accuracy(labels, predictions)

  def write_summary(self, step, mode):
    if mode == 'train':
      loss_metric = self.train_loss
      accuracy_metric = self.train_accuracy
      summary = self.train_summary
    else:
      loss_metric = self.test_loss
      accuracy_metric = self.test_accuracy
      summary = self.test_summary

    with summary.as_default():
      tf.summary.scalar('loss', loss_metric.result(), step=step)
      tf.summary.scalar('accuracy', accuracy_metric.result(), step=step)

  def print_progress(self, epoch, elapse):
    template = 'Epoch {}, Loss {:.4f}, Accuracy: {:.2f}, Test Loss: {:.4f}, ' \
               'Test Accuracy: {:.2f}, Time: {:.2f}s'
    print(
        template.format(epoch, self.train_loss.result(),
                        self.train_accuracy.result() * 100,
                        self.test_loss.result(),
                        self.test_accuracy.result() * 100, elapse))


def get_summary_writer(hparams):
  return tf.summary.create_file_writer(hparams.output_dir)


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

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_datasets as tfds


class HParams(object):
  """Empty object hyper-parameters """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)


class Logger(object):

  def __init__(self):
    self.train_loss = keras.metrics.Mean(name="train_loss")
    self.train_accuracy = keras.metrics.SparseCategoricalAccuracy(
        name="train_accuracy")

    self.test_loss = keras.metrics.Mean(name="test_loss")
    self.test_accuracy = keras.metrics.SparseCategoricalAccuracy(
        name="test_accuracy")

  def log_progress(self, loss, labels, predictions, mode):
    if mode == 'train':
      self.train_loss(loss)
      self.train_accuracy(labels, predictions)
    else:
      self.test_loss(loss)
      self.test_accuracy(labels, predictions)

  def print_progress(self, epoch):
    template = 'Epoch {}, Loss {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(
        template.format(epoch, self.train_loss.result(),
                        self.train_accuracy.result() * 100,
                        self.test_loss.result(),
                        self.test_accuracy.result() * 100))


def get_dataset(hparams):
  dataset, info = tfds.load(
      name=hparams.dataset, with_info=True, as_supervised=True)

  hparams.num_classes = info.features['label'].num_classes

  def scale_image(image, label):
    return tf.cast(image, tf.float32) / 255.0, label

  dataset_train = dataset['train'].map(scale_image).shuffle(10000).batch(
      hparams.batch_size)
  dataset_test = dataset['test'].map(scale_image).batch(hparams.batch_size)

  return dataset_train, dataset_test

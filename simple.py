import os
import tensorflow.keras as keras
from utils import get_hparams, get_dataset, get_optimizer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():
  hparams = get_hparams()
  datasets = get_dataset(hparams)

  model = keras.models.Sequential([
      keras.layers.Flatten(),
      keras.layers.Dense(hparams.num_units, activation='relu'),
      keras.layers.Dropout(hparams.dropout),
      keras.layers.Dense(hparams.num_units, activation='relu'),
      keras.layers.Dropout(hparams.dropout),
      keras.layers.Dense(hparams.num_classes, activation='softmax')
  ])

  optimizer = get_optimizer(hparams)

  model.compile(
      optimizer=optimizer,
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy'])

  model.fit(
      x=datasets['train'],
      y=None,
      epochs=hparams.epochs,
      validation_data=datasets['test'])

  model.summary()


if __name__ == '__main__':
  main()

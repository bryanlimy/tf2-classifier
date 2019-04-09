## Simple classifier with TensorFlow 2.0

### Requirements
```
pip install tensorflow==2.0.0-alpha tensorflow-datasets
```

### Run
- simple sequential model
```
python simple.py
```
- functional model with custom training loop
```
python custom.py
tensorboard --logdir runs
```
- functional model with TensorBoard hparams tuning
```
python tuning.py
tensorboard --logdir runs
```

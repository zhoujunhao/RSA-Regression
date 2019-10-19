# RSA-Regression
Regularization Self-Attention Regression

## Installation
- Python 2.7   
- Tensorflow-gpu 1.5.0  
- Keras 2.1.3
- scikit-learn 0.19

## Train the model
**Run command below to train the model:**
- Train RSA-Regression model based on Gold-price dataset or Palladium-price datset.
```
python RSA-Regression.py
```
You can choose different datasets. Just change the path.

- Train the baseline models based on Gold-price dataset or Palladium-price datset. For example, you can choose LSTM.
```
python LSTM.py
```

## Experiment
Data are obtained from the Caltrans Performance Measurement System (CPeMS) and Fremont Bridge Bicycle Counter (FBBC).
```
device: GTX 1050
OS: Ubuntu 16.04
dataset: Gold-price and Palladium-price
```

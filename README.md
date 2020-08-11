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
You can choose different datasets. Just change the dataset path.

- Train the baseline models based on Gold-price dataset or Palladium-price datset. For example, you can choose LSTM.
```
python LSTM.py
```
## Parameter study
**Run command below to investigate the parameters:**
- Investigate the impacts of parameters based on Gold-price dataset. For example, you can investigate the impact of number of CNN filters. 
```
python RSA-nfilters.py
```

## Experiment
Data are obtained from [Macrotrend](http://www.macrotrends.net). 
```
device: GTX 1050
OS: Ubuntu 16.04
dataset: Gold-price and Palladium-price
```

## Citation
```
@ARTICLE{8943215,
  author={J. {Zhou} and Z. {He} and Y. N. {Song} and H. {Wang} and X. {Yang} and W. {Lian} and H. {Dai}},
  journal={IEEE Access}, 
  title={Precious Metal Price Prediction Based on Deep Regularization Self-Attention Regression}, 
  year={2020},
  volume={8},
  number={},
  pages={2178-2187},}
  ```
  

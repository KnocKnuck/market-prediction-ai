# Python for Finance
# Predict Stock Market Price using Tensorflow 2 and Tensorboard  

Trading Market Prediction using Tensorflow 2 and Keras 
tensorflow pandas numpy matplotlib yahoo finance api and sklearn 

Foreseeing stock costs has consistently been an alluring point to the two speculators and analysts. Speculators consistently question if the cost of a stock will rise or not, since there are many confounded monetary pointers that solitary financial specialists and individuals with great account information can comprehend, the pattern of securities exchange is conflicting and look extremely irregular to customary individuals.

The reason for this personnal project is to construct a neural networks in TensorFlow 2 and Keras that predicts financial exchange costs.  More specifically, we will assemble a Recurrent Neural Network with LSTM cells as it is the present status of-the-craftsmanship in time arrangement forecasting.

To begin with, you need to install those Tensorflow 2 and different libraries:

```
pip3 install tensorflow pandas numpy matplotlib yahoo_fin sklearn requests_html
```

Run train.py in your console 
```
train.py 
```

You can see live results of the learning process using tensorboard and access it at http://localhost:6006/ 
```
tensorboard --logdir="logs"
```

## Output 
```
You are looking for BTC-USD price prediction
Future price after 15 days is 672.37$
huber_loss loss: 3.9528385968878865e-05
Mean Absolute Error: 7.079536935796847
Accuracy score: 0.622093023255814
Total buy profit: 2526.2882380485535
Total sell profit: 148.18811559677124
Total profit: 2674.4763536453247
Profit per trade: 5.183093708614971
```

Graphic showing Actual and Predicted price 
![Image of PythonGraph](https://joseph-hani.fr/img/projet/market-trade/actual-predicted-tensorflow-python-joseph-hani.png)


And finally, it's showing us the last rows of our CSV results : 
```
                  open        high         low       close    adjclose    volume ticker  adjclose_15  true_adjclose_15  buy_profit  sell_profit
2020-10-09  430.130005  434.589996  426.459991  434.000000  434.000000  28925700   TSLA   417.298096        388.040009    0.000000    16.701904
2020-10-13  443.350006  448.890015  436.600006  446.649994  446.649994  34463700   TSLA   417.723969        423.899994    0.000000    28.926025
2020-10-14  449.779999  465.899994  447.350006  461.299988  461.299988  48045400   TSLA   417.543579        420.980011    0.000000    43.756409
2020-10-26  411.630005  425.760010  410.000000  420.279999  420.279999  28239200   TSLA   444.038330        408.089996    0.000000   -23.758331
2020-10-29  409.959991  418.059998  406.459991  410.829987  410.829987  22655300   TSLA   493.665924        499.269989   82.835938     0.000000
2020-11-06  436.100006  436.570007  424.279999  429.950012  429.950012  21706000   TSLA   586.615967        567.599976  156.665955     0.000000
2020-11-17  460.170013  462.000000  433.010010  441.609985  441.609985  61188300   TSLA   629.879822        604.479980  188.269836     0.000000
2020-11-19  492.000000  508.609985  487.570007  499.269989  499.269989  62475300   TSLA   634.758789        609.989990  135.488800     0.000000
2020-11-20  497.989990  502.500000  489.059998  489.609985  489.609985  32807300   TSLA   637.431885        639.830017  147.821899     0.000000
2020-11-30  602.210022  607.799988  554.510010  567.599976  567.599976  63003100   TSLA   660.063904        660.729980   92.463928     0.000000
```


This is tensorboard final train results on mean absolute error :
![Image of Tensorboard1](https://joseph-hani.fr/img/projet/market-trade/tensorboard-python-joseph-hani.png)


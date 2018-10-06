# Regression Channel Algo

conda create --name py35 python=3.5
conda activate py35
conda install spyder
pip install fix_yahoo_finance


Note
The PPP REGRESSION CHANNEL ALGO PROGRAM can be used to trade many assets.
However, if the asset is priced close to zero, the assymetry of the lognormal stock price distribution will become strong
To trade assets close to zero (e.g. penny stocks), take the log of the price before calculating the Z_value indicator.
Taking the log of a lognormal variable converts it to a normal variable.

packages already installed with anaconda:
http://docs.anaconda.com/anaconda/packages/old-pkg-lists/2.5.0/py35/
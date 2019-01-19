# Regression Channel Algo

Note
The Regression Channel Algorithm can be used to trade many assets.
However, if the asset is priced close to zero, the assymetry of the lognormal stock price distribution will become strong
To trade assets close to zero (e.g. penny stocks), take the log of the price before calculating the Z_value indicator.
Taking the log of a lognormal variable converts it to a normal variable. The assets are then traded based on whether the stock has crossed the channel threshold.

Notes: 
   - If prices are not normally distributed this algorithms will not be as effective
   - When crossing the channel we do not accumulate positions (i.e. once we cross the channel we will buy or short the asset and will hold  it until it time to sell)

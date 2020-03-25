from statsmodels.tsa.arima_model import ARIMA


# Function that calls ARIMA model to fit
def arima_forecast(Actual, P, D, Q):
    model = ARIMA(Actual, order=(P, D, Q))
    model_fit = model.fit(disp=0)
    print(model_fit.summary())
    prediction = model_fit.forecast()[0]
    return prediction

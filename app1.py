import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model as lm
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler as mms
from sklearn.metrics import r2_score as r2s
from sklearn.metrics import mean_squared_error as mse
from shapely.geometry import LineString as ls


st.header('Stock Market Predictor')

stock = st.text_input('Enter stock symbol', 'GOOG')

start = '1989-01-01'
end = '2023-12-31'

data = yf.download(stock, start, end)

data.reset_index(inplace=True)
st.subheader('STOCK DATA')
st.write(data)

#variables
win_size1=10
win_size=2
mva_50 = data.Close.rolling(50).mean()
mva_200 = data.Close.rolling(200).mean()
arr_mva=np.array(mva_200)
scaler = mms(feature_range=(0, 1))



# model based on closing price
data_train = pd.DataFrame(data.Close[0: int(len(data) * 0.80)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80): len(data)])



pst_100_days = data_train.tail(100)
data_test = pd.concat([pst_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)


st.subheader('Moving Average 200 days vs 50 days vs Original Closing Price')
fig2 = plt.figure(figsize=(10, 6))
plt.plot(mva_50, 'r', label='Moving Average of 50 days')
plt.plot(mva_200, 'y', label='Moving Average of 200 days')
plt.plot(data.Close, 'g', label='Closing price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i, 0])

x, y = np.array(x), np.array(y)

model = lm(r"Stock Prediction Model.keras")

predict = model.predict(x)
scale = 1/scaler.scale_
predict = predict * scale
y = y * scale

st.subheader('Original price vs Predicted price')
fig = plt.figure(figsize=(10, 6))
plt.plot(predict, 'r', label='Predicted price')
plt.plot(y, 'g', label='Original price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)

r2 = r2s(y, predict)
mean_error = np.sqrt(mse(y, predict))

st.subheader('The r2 score of the model is')
st.write(r2)

st.subheader('The mean squared error of our model is')
st.write(mean_error)

#MVA section
data_train_mva = pd.DataFrame(mva_200[0: int(len(mva_200) * 0.80)])
data_test_mva = pd.DataFrame(mva_200[int(len(mva_200) * 0.80): len(mva_200)])

pst_days_mva = data_train_mva.tail(win_size1)
data_test_mva = pd.concat([pst_days_mva, data_test_mva], ignore_index=True)
data_test_mva_scale = scaler.fit_transform(data_test_mva)

x1 = []
y1 = []

for i in range(win_size1, data_test_mva_scale.shape[0]):
    x1.append(data_test_mva_scale[i-win_size1:i])
    y1.append(data_test_mva_scale[i, 0])

x1, y1 = np.array(x1), np.array(y1)


model1=lm(r'Stock Predictions Model longterm.keras')

predict_mva = model1.predict(x1)
predict_mva = predict_mva * scale
y1 = y1 * scale


st.subheader('Original MVA_200 Price vs Predicted price')
fig = plt.figure(figsize=(10, 6))
plt.plot(predict_mva, 'r', label='Predicted price')
plt.plot(y1, 'g', label='Original price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)

r2_mva = r2s(y1, predict_mva)
mean_error_mva = (np.sqrt(mse(y1, predict_mva)))

st.subheader('The r2 score of the model is')
st.write(r2_mva)

st.subheader('The mean squared error of our model is')
st.write(mean_error_mva)





#tops  and bottoms
def directional_change(close: np.array, high: np.array, low: np.array, sigma: float):
    
    up_zig = True # Last extreme is a bottom. Next is a top. 
    tmp_max = high[0]
    tmp_min = low[0]
    tmp_max_i = 0
    tmp_min_i = 0

    tops = []
    bottoms = []

    for i in range(len(close)):
        if up_zig: # Last extreme is a bottom
            if high[i] > tmp_max:
                # New high, update 
                tmp_max = high[i]
                tmp_max_i = i
            elif close[i] < tmp_max - tmp_max * sigma: 
                # Price retraced by sigma %. Top confirmed, record it
                # top[0] = confirmation index
                # top[1] = index of top
                # top[2] = price of top
                top = [i, tmp_max_i, tmp_max]
                tops.append(top)

                # Setup for next bottom
                up_zig = False
                tmp_min = low[i]
                tmp_min_i = i
        else: # Last extreme is a top
            if low[i] < tmp_min:
                # New low, update 
                tmp_min = low[i]
                tmp_min_i = i
            elif close[i] > tmp_min + tmp_min * sigma: 
                # Price retraced by sigma %. Bottom confirmed, record it
                # bottom[0] = confirmation index
                # bottom[1] = index of bottom
                # bottom[2] = price of bottom
                bottom = [i, tmp_min_i, tmp_min]
                bottoms.append(bottom)

                # Setup for next top
                up_zig = True
                tmp_max = high[i]
                tmp_max_i = i

    return tops, bottoms

tops1,bottoms1=directional_change(arr_mva,data.High,data.Low,0.01)    # MVA OR ORIGIANAL CHANGES

arr_tops1=[]
arr_bottoms1=[]
for i in range(len(tops1)):
    arr_tops1.append(tops1[i][2])

for i in range(len(bottoms1)):
    arr_bottoms1.append(bottoms1[i][2])
    

extremes=np.concatenate((arr_tops1,arr_bottoms1))

data_train_extreme = pd.DataFrame(extremes[0: int(len(extremes)*0.80)])
data_test_extreme = pd.DataFrame(extremes[int(len(extremes)*0.80): len(extremes)])

pst_days_extreme = data_train_extreme.tail(win_size)
data_test_extreme = pd.concat([pst_days_extreme, data_test_extreme], ignore_index=True)
data_test_extreme_scale = scaler.fit_transform(data_test_extreme)

x2 = []
y2 = []

for i in range(win_size, data_test_extreme_scale.shape[0]):
    x2.append(data_test_extreme_scale[i-win_size:i])
    y2.append(data_test_extreme_scale[i, 0])

x2, y2 = np.array(x2), np.array(y2)


model2=lm(r'Stock top_down1.keras')

predict_extreme = model2.predict(x2)
predict_extreme = predict_extreme * scale
y2 = y2 * scale


st.subheader('Tops And Bottoms of original MVA_200 and Predicted')
fig = plt.figure(figsize=(10, 6))
plt.plot(predict_extreme, 'r', label='Predicted price')
plt.plot(y2, 'g', label='Original price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)


r2_extreme = r2s(y2, predict_extreme)
mean_error_extreme = (np.sqrt(mse(y2, predict_extreme)))

st.subheader('The r2 score of the model is')
st.write(r2_extreme)

st.subheader('The mean squared error of our model is')
st.write(mean_error_extreme)



#days

arr_tops1=[]
arr_bottoms1=[]
for i in range(len(tops1)):
    arr_tops1.append(tops1[i][0])

for i in range(len(bottoms1)):
    arr_bottoms1.append(bottoms1[i][0])

days_between1 = []
if tops1[0][1]>bottoms1[0][1]:
    days_between1.append(bottoms1[0][1])
    for i in range(len(tops1)):
        days_between1.append(abs(arr_bottoms1[i]-arr_tops1[i]))
        days_between1.append(abs(arr_bottoms1[i]-arr_tops1[i-1]))

elif bottoms1[0][1]>tops1[0][1]:
    days_between1.append(bottoms1[0][1])
    for i in range(len(bottoms1)):
        days_between1.append(abs(arr_tops1[i]-arr_bottoms1[i]))
        days_between1.append(abs(arr_tops1[i]-arr_bottoms1[i-1]))
        
        
print(days_between1)
print(len(days_between1))
print(np.mean(days_between1))
print(np.std(days_between1))

count=0
day=np.array(days_between1)
for i in range(len(day)):
    if day[i]<1:
        count+=1
print(count)

# days_between1.pop(2)


data_train_days = pd.DataFrame(days_between1[0: int(len(days_between1)*0.80)])
data_test_days = pd.DataFrame(days_between1[int(len(days_between1)*0.80): len(days_between1)])

pst_days_days = data_train_days.tail(win_size)
data_test_days = pd.concat([pst_days_days, data_test_days], ignore_index=True)
data_test_days_scale = scaler.fit_transform(data_test_days)

x3 = []
y3 = []

for i in range(win_size, data_test_days_scale.shape[0]):
    x3.append(data_test_days_scale[i-win_size:i])
    y3.append(data_test_days_scale[i, 0])

x3, y3 = np.array(x3), np.array(y3)


model3=lm(r'Stock days1.keras')

predict_days = model3.predict(x3)
predict_days = predict_days * scale
y3 = y3 * scale


st.subheader('Tops And Bottoms of original MVA_200 and Predicted')
fig = plt.figure(figsize=(10, 6))
plt.plot(predict_days, 'r', label='Predicted price')
plt.plot(y3, 'g', label='Original price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)

r2_days = r2s(y3, predict_days)
mean_error_days = (np.sqrt(mse(y3, predict_days)))

st.subheader('The r2 score of the model is')
st.write(r2_days)

st.subheader('The mean squared error of our model is')
st.write(mean_error_days)


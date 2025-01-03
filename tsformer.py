import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import time

prices = pd.read_csv("thyao.csv")
prices['Tarih'] = pd.to_datetime(prices['Tarih'])
prices['Kapanış Fiyatı'] = prices['Kapanış Fiyatı'].astype(float)
prices = prices.interpolate()

cutoff_date = prices['Tarih'].max() - pd.DateOffset(years=2)
prices_recent = prices[prices['Tarih'] >= cutoff_date].copy()

prices_recent['Day'] = prices_recent['Tarih'].dt.day
prices_recent['Month'] = prices_recent['Tarih'].dt.month
prices_recent['Year'] = prices_recent['Tarih'].dt.year
prices_recent['Weekday'] = prices_recent['Tarih'].dt.weekday
prices_recent['Days'] = (prices_recent['Tarih'] - prices_recent['Tarih'].min()).dt.days
prices_recent['7D_MA'] = prices_recent['Kapanış Fiyatı'].rolling(window=7, min_periods=1).mean()
prices_recent['30D_MA'] = prices_recent['Kapanış Fiyatı'].rolling(window=30, min_periods=1).mean()
scaler = MinMaxScaler()
prices_recent['Scaled Fiyat'] = scaler.fit_transform(prices_recent[['Kapanış Fiyatı']])

X_prices = prices_recent[['Days', 'Day', 'Month', 'Year', 'Weekday', '7D_MA', '30D_MA']].fillna(0).values
y_prices = prices_recent['Scaled Fiyat'].values

train_cutoff = prices_recent['Tarih'].max() - pd.DateOffset(months=3)
train_data = prices_recent[prices_recent['Tarih'] <= train_cutoff].copy()
test_data = prices_recent[prices_recent['Tarih'] > train_cutoff].copy()

X_train = train_data[['Days', 'Day', 'Month', 'Year', 'Weekday', '7D_MA', '30D_MA']].fillna(0).values
y_train = train_data['Scaled Fiyat'].values
X_test = test_data[['Days', 'Day', 'Month', 'Year', 'Weekday', '7D_MA', '30D_MA']].fillna(0).values
y_test = test_data['Scaled Fiyat'].values

class TSformerModel(nn.Module):
    def __init__(self, input_size=7, hidden_size=128, output_size=1, num_layers=3, dropout=0.4):
        super(TSformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, dropout=dropout)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  
        out = self.transformer(x)
        out = out[-1, :, :]
        out = self.fc(out)
        return out

input_dim = 7
hidden_dim = 128
output_dim = 1

tsformer_model = TSformerModel(input_size=input_dim, hidden_size=hidden_dim, output_size=output_dim, num_layers=3, dropout=0.4)
criterion = nn.SmoothL1Loss()
tsformer_optimizer = torch.optim.AdamW(tsformer_model.parameters(), lr=0.00001, weight_decay=3e-5)
epochs = 200

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

tsformer_model.train()
tsformer_train_losses = []
tsformer_test_losses = []
training_start_time = time.time()
for epoch in range(epochs):
    tsformer_optimizer.zero_grad()
    tsformer_outputs = tsformer_model(X_train_tensor.unsqueeze(1))
    tsformer_loss = criterion(tsformer_outputs, y_train_tensor)
    tsformer_loss.backward()
    tsformer_optimizer.step()
    tsformer_train_losses.append(tsformer_loss.item())

    tsformer_model.eval()
    with torch.no_grad():
        tsformer_test_outputs = tsformer_model(X_test_tensor.unsqueeze(1))
        tsformer_test_loss = criterion(tsformer_test_outputs, y_test_tensor)
        tsformer_test_losses.append(tsformer_test_loss.item())
    tsformer_model.train()
training_end_time = time.time()
training_time = training_end_time - training_start_time

plt.figure(figsize=(10, 6))
plt.plot(range(epochs), tsformer_train_losses, label='Eğitim Kaybı', color='blue')
plt.plot(range(epochs), tsformer_test_losses, label='Test Kaybı', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Epoch vs Loss (Eğitim ve Test)')
plt.legend()
plt.grid(True)
plt.show()

future_days = 30
last_day = prices_recent['Days'].max()
future_inputs = np.array([[last_day + i, (last_day + i) % 31, ((last_day + i) // 31) % 12 + 1, 2024, (last_day + i) % 7, 0, 0] for i in range(1, future_days + 1)])
future_inputs_tensor = torch.tensor(future_inputs, dtype=torch.float32)

inference_start_time = time.time()
tsformer_model.eval()
tsformer_future_predictions = tsformer_model(future_inputs_tensor.unsqueeze(1)).detach().numpy().squeeze(-1)
tsformer_future_predictions = scaler.inverse_transform(tsformer_future_predictions.reshape(-1, 1)).flatten()
inference_end_time = time.time()
inference_time = inference_end_time - inference_start_time

plt.figure(figsize=(12, 8))
plt.plot(prices_recent['Tarih'], scaler.inverse_transform(prices_recent[['Scaled Fiyat']]), label="Geçmiş Fiyatlar", color="gray", linestyle="dotted")
plt.plot(test_data['Tarih'], scaler.inverse_transform(test_data[['Scaled Fiyat']]), label="Gerçek Değerler", color="red")

future_dates = pd.date_range(start=prices_recent['Tarih'].max() + pd.Timedelta(days=1), periods=future_days)
plt.plot(future_dates, tsformer_future_predictions, label="TSformer Tahminleri", linestyle="dashed", color="blue")

plt.legend()
plt.xlabel("Tarih")
plt.ylabel("Fiyat")
plt.title("TSformer Tahmin Grafiği")
plt.grid(True)
plt.show()

y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))[:len(tsformer_future_predictions)]  
mse = mean_squared_error(y_test_actual, tsformer_future_predictions)
mape = np.mean(np.abs((y_test_actual.flatten() - tsformer_future_predictions) / y_test_actual.flatten())) * 100
mae = mean_absolute_error(y_test_actual, tsformer_future_predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_actual, tsformer_future_predictions)

print("TSformer Model Performansı:")
print(f"MSE: {mse}")
print(f"MAPE: {mape}")
print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"R-Squared: {r2}")
print(f"Eğitim Süresi: {training_time:.2f} saniye")
print(f"Çıkarım Süresi: {inference_time:.2f} saniye")

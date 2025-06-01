import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import Sequential # from tensorflow.keras.models import Sequential
from keras import layers # from tensorflow.keras.layers import Dense
from keras import losses # from tensorflow.keras.losses import MeanSquaredError
from sklearn.model_selection import train_test_split
import joblib

# Загрузка и подготовка данных
df = pd.read_csv('dataset/tambov_weather.csv')

df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

# Входные признаки
x = df[['year', 'month', 'day']].values

# Выходные признаки
y = df[['avg_temp_c', 'min_temp_c', 'max_temp_c', 'precipitation_mm']].values

# Масштабирование выходов
x_scaler = MinMaxScaler()
x_scaled = x_scaler.fit_transform(x)
y_scaler = MinMaxScaler()
y_scaled = y_scaler.fit_transform(y)

# Разделение на выборки
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.2, random_state=42)

# Создание и обучение модели
model = Sequential([
    layers.Dense(128, activation='relu', input_shape=(x_train.shape[1],)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(4)
])
model.compile(optimizer='adam', loss=losses.MeanSquaredError(), metrics=['mae'])

history = model.fit(x_train, y_train, epochs=50, batch_size=16, validation_split=0.1)

loss = model.evaluate(x_test, y_test)
print(f'Test MSE: {loss}')

from matplotlib.axes import Axes
# График потерь
fig, axs = plt.subplots(2, 1, figsize=(10, 10))
f1: Axes = axs[0]
f2: Axes = axs[1]

#fig.tight_layout(h_pad=4)
f1.set_yscale('log')
f2.set_yscale('log')

f1.plot(history.history['loss'], label='train')
f1.plot(history.history['val_loss'], label='val')
f1.set_title('Loss over epochs')
f1.set_xlabel('Epoch')
f1.set_ylabel('MSE Loss')
f1.legend()

f2.plot(history.history['mae'], label='train')
f2.plot(history.history['val_mae'], label='val')
f2.set_title('MAE over epochs')
f2.set_xlabel('Epoch')
f2.set_ylabel('MAE')
f2.legend()

#fig.savefig('plot.png')
#os.startfile('plot.png')
fig.show()
quit()
# Сохранение масштабатора
joblib.dump(x_scaler, 'models/x_scaler.save')
joblib.dump(y_scaler, 'models/y_scaler.save')
# Сохранение модели
model.save('models/weather_model.keras')
print("Модель и масштабатор сохранены.")


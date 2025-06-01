import numpy as np
from tensorflow.keras.models import load_model
import joblib
from datetime import datetime

# Загрузка модели и масштабаторов
model = load_model('models/weather_model.keras')
x_scaler = joblib.load('models/x_scaler.save')
y_scaler = joblib.load('models/y_scaler.save')

# Ввод даты
input_date_str = input('Введите дату в формате YYYY-mm-dd\n> ') #"2025-07-24"
input_date = datetime.strptime(input_date_str, "%Y-%m-%d")
x_input = np.array([[input_date.year, input_date.month, input_date.day]])

# Масштабирование входа
x_scaled = x_scaler.transform(x_input)

# Предсказание и обратное масштабирование
pred_scaled = model.predict(x_scaled)
prediction = y_scaler.inverse_transform(pred_scaled)[0]

# Названия выходных параметров
columns = ['avg_temp_c', 'min_temp_c', 'max_temp_c', 'precipitation_mm']

# Вывод
print(f"\nПрогноз погоды на {input_date_str}:")
for name, value in zip(columns, prediction):
    print(f"{name}: {value:.2f}")

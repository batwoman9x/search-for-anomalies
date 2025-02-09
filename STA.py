import os
import gzip
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
import plotly.graph_objects as go
import urllib.request

# URL для загрузки
url = "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz"
file_path = "kddcup.data_10_percent.gz"

# Проверяем наличие файла, если нет — скачиваем
if not os.path.exists(file_path):
    print(f"Файл {file_path} не найден. Скачиваем из {url}...")
    urllib.request.urlretrieve(url, file_path)
    print(f"Файл {file_path} успешно загружен.")

# Имена столбцов
column_names = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type', 'level'
]

# Загрузка данных
with gzip.open(file_path, 'rt') as f:
    data = pd.read_csv(f, header=None, names=column_names)

# Предварительная обработка данных
categorical_columns = ['protocol_type', 'service', 'flag']
data = pd.get_dummies(data, columns=categorical_columns)

# Разделение на признаки и метки
X = data.drop(['attack_type', 'level'], axis=1)
y = data['attack_type']

# Преобразование меток атак в бинарные (нормальный трафик и нет)
y = y.apply(lambda x: 0 if x == 'normal.' else 1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Реформатирование для LSTM
X_scaled = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))

# Инициализация кросс-валидации Stratified K-Fold
skf = StratifiedKFold(n_splits=5)

accuracies = []
fold_no = 1

# Проверка устройства для TensorFlow
print("Используемое устройство для выполнения:", tf.config.list_physical_devices('GPU') or "CPU")

for train_index, test_index in skf.split(X_scaled, y):
    print(f"Training on fold {fold_no}...")

    # Разделение данных на тренировочные и тестовые
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Создание модели LSTM
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))  # Используем Input вместо input_shape
    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Бинарная классификация

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Обучение 
    history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2, verbose=0)

    # Оценка 
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    assert len(y_pred) == len(y_test), "Размеры предсказаний и меток не совпадают!"
    
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

    print(f"Accuracy for fold {fold_no}: {accuracy}")
    fold_no += 1

# Средняя точность
mean_accuracy = np.mean(accuracies)
print(f"Mean Accuracy across all folds: {mean_accuracy}")

# Визуализация угроз во времени с использованием Plotly
time = np.arange(len(y_pred))
threat_intensity = X_test[:, 0, 1]  

# Построение 3D-графика
fig = go.Figure(data=[go.Scatter3d(
    x=time,
    y=threat_intensity,
    z=y_pred.flatten(),
    mode='markers',
    marker=dict(
        size=5,
        color=y_pred.flatten(), 
        colorscale='Viridis',
        opacity=0.8
    )
)])

# Настройки осей и отображения
fig.update_layout(scene=dict(
                    xaxis_title='Time',
                    yaxis_title='Threat Intensity',
                    zaxis_title='Threat Type'),
                  margin=dict(r=20, b=10, l=10, t=10))

fig.show()
# Сохранение файла
fig.write_html("output_graph.html")

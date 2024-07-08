import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
import matplotlib.pyplot as plt
from keras import regularizers
import joblib  


df = pd.read_csv("/Applications/XAMPP/xamppfiles/htdocs/PEagle/flask/our_db_5.csv")
df.head()

X = df[['mark', 'name', 'km_driven', 'year']]
y = df['price']

X = pd.get_dummies(X, columns=['mark', 'name'])

scaler_km = RobustScaler()
scaler_year = RobustScaler()

X[['km_driven']] = scaler_km.fit_transform(X[['km_driven']])
X[['year']] = scaler_year.fit_transform(X[['year']])

joblib.dump(scaler_km, '/Applications/XAMPP/xamppfiles/htdocs/PEagle/flask/def_scaler_km.pkl')
joblib.dump(scaler_year, '/Applications/XAMPP/xamppfiles/htdocs/PEagle/flask/def_scaler_year.pkl')


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)
X_train = X_train.astype('float32')
X_test= X_test.astype('float32')
model = keras.Sequential([
    keras.layers.Dense(256, input_dim=X_train.shape[1], activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(256, activation='relu',kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dropout(0.5),   
    keras.layers.Dense(1)
])


model.compile(loss='mean_squared_error', optimizer='adam')

# Durante l'addestramento, aggiungi ModelCheckpoint alla lista dei callbacks
history = model.fit(X_train, y_train, epochs=10, batch_size=2, validation_data=(X_test, y_test))


model.save("/Applications/XAMPP/xamppfiles/htdocs/PEagle/flask/def_model.keras")
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import joblib

datos = pd.read_excel('Dataset.xlsx',header=0)

celsiu = datos['Celsius']
fahrenheit = datos['Fahrenheit']


oculta1 = tf.keras.layers.Dense(units=4,input_shape=[1])
oculta2 = tf.keras.layers.Dense(units=4)
salida = tf.keras.layers.Dense(units=1)
modelo = tf.keras.Sequential([oculta1,oculta2,salida])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
    )

print('Comenzando entrenamiento ....')
historial = modelo.fit(celsiu,fahrenheit,epochs=1000, verbose=False)
print('Modelo entrenado')

plt.xlabel("# perdida")
plt.ylabel("Magnitud de perdida")
plt.plot(historial.history["loss"])

resultado = modelo.predict([640])
print(resultado)




joblib.dump(modelo,'modelo_entrenado.pkl')
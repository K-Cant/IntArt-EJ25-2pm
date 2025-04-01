# Imports necesarios
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

# cargamos los datos de entrada
data = pd.read_csv("./articulos_ml.csv")

# filtrar los datos
filtered_data = data[(data['Word count'] <= 3500) & (data['# Shares'] <= 80000)]

# variable nueva = suma de los enlaces, comentarios e imagenes
suma = (filtered_data['# of Links'] + filtered_data['# of comments'].fillna(0) + filtered_data['# Images video'])

# preparación de las variables antes del entrenamiento
dataX2 = pd.DataFrame()
dataX2['Word count'] = filtered_data['Word count']
dataX2['suma'] = suma
XY_train = np.array(dataX2)
z_train = filtered_data['# Shares'].values

regr2 = linear_model.LinearRegression() # objeto de regresion lineal
regr2.fit(XY_train, z_train) # entrenar el modelo con 2 dimensiones
z_pred = regr2.predict(XY_train) # prediccion

print('Coefficients: ', regr2.coef_)
print("Mean squared error: %.2f" % mean_squared_error(z_train, z_pred))
print('Variance score: %.2f' % r2_score(z_train, z_pred))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d') # graficar 3D

# malla sobre la cual se graficara el plano
xx, yy = np.meshgrid(np.linspace(0, 3500, num=10), np.linspace(0, 60, num=10))

# calculo de los valores del plano
nuevoX = (regr2.coef_[0] * xx)
nuevoY = (regr2.coef_[1] * yy)
z = (nuevoX + nuevoY + regr2.intercept_)

ax.plot_surface(xx, yy, z, alpha=0.2, cmap='hot') # graficamos el plano
ax.scatter(XY_train[:, 0], XY_train[:, 1], z_train, c='blue',s=30) # azul los puntos de entrenamiento
ax.scatter(XY_train[:, 0], XY_train[:, 1], z_pred, c='red',s=40) # rojo los puntos predictivos
ax.view_init(elev=30., azim=65) # situamos la 'camara' para visualizar 

ax.set_xlabel('Cantidad de Palabras')
ax.set_ylabel('Cantidad de Enlaces,Comentarios e Imagenes')
ax.set_zlabel('Compartido en Redes')
ax.set_title('Regresión Lineal con Múltiples Variables')

plt.show()

# predecir cuantos #Shares voy a obtener por un artículo con:
# 2000 palabras, 10 enlaces, 4 comentarios, 6 imagenes
z_Dosmil = regr2.predict([[2000, 10+4+6]])
print('Predicción: ', int(z_Dosmil))
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

print(data.shape) # dimensiones y registros
print("\n", data.head()) # primeros registros
print("\n", data.describe()) # estadísticas de los datos

# visualización en histograma
data.drop(['Title','url', 'Elapsed days'],axis=1).hist()
plt.show()

# filtrar los datos
filtered_data = data[(data['Word count'] <= 3500) & (data['# Shares'] <= 80000)]

# pintar en colores los puntos por debajo y por encima de la media
colores=['orange','blue']
tamanios=[30,60]

f1 = filtered_data['Word count'].values
f2 = filtered_data['# Shares'].values

asignar = []
for index, row in filtered_data.iterrows():
    if(row['Word count']>1808):
        asignar.append(colores[0])
    else:
        asignar.append(colores[1])

plt.scatter(f1, f2, c=asignar, s=tamanios[0])
plt.show()

# etiquetar los datos para entrenamiento
dataX = filtered_data[["Word count"]]
X_train = np.array(dataX)
y_train = filtered_data['# Shares'].values

regr = linear_model.LinearRegression() # objeto de regresión linear
regr.fit(X_train, y_train) # entrenar el modelo
y_pred = regr.predict(X_train) # predicciones

print('\nCoeficientes: ', regr.coef_) 
print('Terminos independientes: ', regr.intercept_)
print("Error cuadrado medio: %.2f" % mean_squared_error(y_train, y_pred))
print("Puntaje de Varianza: %.2f\n" % r2_score(y_train, y_pred))

# Predicción usando el algoritmo
y_Dosmil = regr.predict([[2000]])
print("Predicción: ", int(y_Dosmil[0]))

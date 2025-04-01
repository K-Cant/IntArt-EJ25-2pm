import pandas as pd
import numpy as np
from sklearn import linear_model, model_selection
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sb

# carga y lectura del archivo CSV
dataframe = pd.read_csv(r"usuarios_win_mac_lin.csv")

print(dataframe.head()) # primeros registros
print("\n", dataframe.describe()) # estadisticas de los datos
print("\n", dataframe.groupby('clase').size()) # tipos de datos

# visualizacion de datos
dataframe.drop(['clase'], axis = 1).hist()
plt.show()

# interrelacionar para ver la relacion lineal 
sb.pairplot(dataframe.dropna(), hue='clase', height=4, vars=["duracion", "paginas", "acciones", "valor"], kind="reg")
plt.show()

X = np.array(dataframe.drop(['clase'],axis=1))
y = np.array(dataframe['clase'])
print("\n", X.shape) # dimension de la matriz

model = linear_model.LogisticRegression(max_iter=1000) # objeto de regresion logistica
model.fit(X,y) # ajustamos el modelo
predictions = model.predict(X) # predicciones

print("\n", predictions[0:5]) # revisamos algunas salidas
print("\n", model.score(X,y)) # revisamos la presicion media de las predicciones

# validamos el modelo dividiendo los datos 80, 20
validation_size = 0.20 
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)

# compilamos el modelo pero solo con el 80 de los datos
name='Logistic Regression'
kfold = model_selection.KFold(n_splits=10, random_state=None)
cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
print("\n", msg) # calculamos nueva presición media

predictions = model.predict(X_validation) # predicciones c
print("\n", accuracy_score(Y_validation, predictions)) # presicion de los datos
print("\n", confusion_matrix(Y_validation, predictions)) # reporte de matriz
print("\n", classification_report(Y_validation, predictions)) # reporte de clasificacion

# clasificacion de nuevos valores
X_new = pd.DataFrame({'duracion': [10], 'paginas': [3], 'acciones': [5], 'valor': [9]}) # nuevos valores
prediction = model.predict(X_new.values) # predecimos
print("\n", "Prediccion: ", prediction) # prediccion
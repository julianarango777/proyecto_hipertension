#Carga de las librerias necesarias

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import seaborn as sb
 
#Carga de funciones necesarias
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


#Carga de datos
db=pd.read_csv('/Users/Julian/Desktop/diabetes/diabetes_2.0.csv')
db2=pd.read_csv('/Users/Julian/Desktop/diabetes/db.csv')
db_o=pd.read_csv('/Users/Julian/Desktop/diabetes/db.csv')

#Ver nombres de columnas
for col in db.columns: 
    print(col) 
    
#Modificaciones necesarias en la base de datos original (db)

db.rename(columns = {list(db)[0]: 'Edad'}, inplace = True)
db.rename(columns={'Grupo de edad':'Gedad'}, inplace=True)
db.drop('Gedad', axis=1, inplace = True)
db.drop('Genero', axis=1, inplace = True)
db.drop('Etnia', axis=1, inplace = True)
db.drop('Afiliacion SGSSS', axis=1, inplace = True)
db.drop('Escolaridad', axis=1, inplace = True)
db.drop('Zona ', axis=1, inplace = True)
db.drop('Discapacidad', axis=1, inplace = True)
db.rename(columns = {list(db)[1]: 'regimen afiliacion'}, inplace = True)
db.drop('regimen afiliacion', axis=1, inplace = True)
db.drop('Fecha Control ', axis=1, inplace = True)
db.drop('Fumador Activo', axis=1, inplace = True)
db.rename(columns = {list(db)[1]: 'indicador diabetes'}, inplace = True)
db.rename(columns = {list(db)[2]: 'Etiqueta Hipertension'}, inplace = True)
db.rename(columns = {list(db)[14]: 'Rframingham'}, inplace = True)
db.rename(columns = {list(db)[15]: 'RCV Global'}, inplace = True)
db.rename(columns = {list(db)[14]: 'RCV Global'}, inplace = True)
db.rename(columns = {list(db)[22]: 'Clasif perimetro abdominal'}, inplace = True)
db.rename(columns = {list(db)[28]: 'Factor correcion formula'}, inplace = True)
db.drop('Clasificacion de Diabetes o del ultimo estado de Glicemia', axis=1, inplace = True)
db.drop('Clasif perimetro abdominal', axis=1, inplace = True)
db.drop('Observaciones', axis=1, inplace = True)
db.drop('DM COMPENSADO', axis=1, inplace = True)
db.drop('HTA Y DM COMPENSADA', axis=1, inplace = True)
db.drop('Antecedentes Fliar  Enfermedad Coronaria', axis=1, inplace = True)
db.drop('CLAIFICACION IMC', axis=1, inplace = True)
db.drop('Remisiones Especialidad', axis=1, inplace = True)
db.drop('Complicaciones  y Lesiones en Organo Blanco', axis=1, inplace = True)
db.drop('HTA COMPENSADOS', axis=1, inplace = True)
db.drop('Perimetro Abdominal', axis=1, inplace = True)
db.drop('Talla', axis=1, inplace = True)
db.drop(list(db)[23], axis=1, inplace = True)
db.drop(list(db)[3], axis=1, inplace = True)
db.drop('Microalbuminuria ', axis=1, inplace = True)
db.drop('indicador diabetes', axis=1, inplace = True)
db.rename(columns = {list(db)[11]: 'Glicemia PPD'}, inplace = True)
db.drop('Glicemia PPD', axis=1, inplace = True)
db.rename(columns = {list(db)[18]: 'Antidiabeticos'}, inplace = True)
db.drop(list(db)[16], axis=1, inplace = True)
db.drop('Estadio IRC', axis=1, inplace = True)
db.rename(columns = {list(db)[1]: 'label'}, inplace = True)
db.drop('Unnamed: 0', axis=1, inplace = True)

db = db[np.isfinite(db['Edad'])]
db = db[np.isfinite(db['Tension SISTOLICA'])]
db = db[np.isfinite(db['Tension DIASTOLICA'])]
db = db[np.isfinite(db['Colesterol Total'])]
db = db[np.isfinite(db['Colesterol HDL'])]
db = db[np.isfinite(db['Trigliceridos'])]
db = db[np.isfinite(db['Glicemia de ayuno'])]
db = db[np.isfinite(db['Creatinina'])]
db = db[np.isfinite(db['Estatina'])]


#Cantidades de valores nulos por columnas
db.isnull().sum()


#Quita valores nulos en filas seleccionadas
db = db[np.isfinite(db['Creatinina'])]

#
dataframe=db
dataframe.head(10)
X = dataframe[['Edad','Tension SISTOLICA','Tension DIASTOLICA','Colesterol Total','Colesterol HDL','Trigliceridos','Glicemia de ayuno','Creatinina']].values
y = dataframe['label'].values
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Definicón de número de vecinos
n_neighbors = 7

#Clasificador y rendimientos

knn = KNeighborsClassifier(n_neighbors)
knn.fit(X_train, y_train)
print('Exactitud del clasificador K-NN  en el conjunto de entrenamiento: {:.2f}'
     .format(knn.score(X_train, y_train)))
print('Exactitud del clasificador K-NN  en el conjunto de evaluacion: {:.2f}'
     .format(knn.score(X_test, y_test)))




#Creación de matriz de confución  y reporte de clasificacion
pred = knn.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))



#Función para clasificar con base a los vecinos más cercanos
clf = KNeighborsClassifier(n_neighbors, weights='distance')
clf.fit(X, y)


#Predicción y probabilidad -> Estos son los parametros a ingresar (Dejo un ejemplo) ['Edad','Tension SISTOLICA','Tension DIASTOLICA','Colesterol Total','Colesterol HDL','Trigliceridos','Glicemia de ayuno','Creatinina']
print('La clasificacion de la muestra es:', (clf.predict([[35, 100, 80, 0.5, 0, 0.5, 0.1, 0.2 ]])))
print('La probabilidad de la muestra es:', (clf.predict_proba([[35, 100, 80, 0.5, 0, 0.5, 0.1, 0.2 ]])))





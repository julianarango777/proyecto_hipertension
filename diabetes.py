import pandas as pd
import numpy as np

#Carga de la base de datos
db=pd.read_csv('/Users/Julian/Desktop/diabetes/diabetes.csv')



#Ver nombres de columnas
for col in db.columns: 
    print(col) 
    
#renombrar columnas con defectos de escritura
db.rename(columns = {list(db)[0]: 'Edad'}, inplace = True)
db.rename(columns={'Grupo de edad':'Gedad'}, inplace=True)
db.rename(columns = {list(db)[1]: 'regimen afiliacion'}, inplace = True)
db.rename(columns = {list(db)[1]: 'indicador diabetes'}, inplace = True)
db.rename(columns = {list(db)[2]: 'Etiqueta Hipertension'}, inplace = True)
db.rename(columns = {list(db)[13]: 'Rframingham'}, inplace = True)
db.rename(columns = {list(db)[14]: 'RCV Global'}, inplace = True)
db.rename(columns = {list(db)[21]: 'Clasif perimetro abdominal'}, inplace = True)
db.rename(columns = {list(db)[22]: 'Factor correcion formula'}, inplace = True)
db.rename(columns = {list(db)[29]: 'inscripcion acciones'}, inplace = True)
db.rename(columns = {list(db)[27]: 'Antidiabeticos'}, inplace = True)
db.rename(columns = {list(db)[24]: 'Farmacos Antihipertensivos'}, inplace = True)


#borrar columnas innecesarias
db.drop('Afiliacion SGSSS', axis=1, inplace = True)
db.drop('Grupo de edad', axis=1, inplace = True)
db.drop('Escolaridad', axis=1, inplace = True)
db.drop('Genero', axis=1, inplace = True)
db.drop('Etnia', axis=1, inplace = True)
db.drop('Discapacidad', axis=1, inplace = True)
db.drop('Fumador Activo', axis=1, inplace = True)
db.drop('zona', axis=1, inplace = True)
db.drop('regimen afiliacion', axis=1, inplace = True)
db.drop('fecha_control', axis=1, inplace = True)
db.drop('Clasificacion de Diabetes o del ultimo estado de Glicemia', axis=1, inplace = True)
db.drop('Clasif perimetro abdominal', axis=1, inplace = True)
db.drop('Observaciones', axis=1, inplace = True)
db.drop('DM COMPENSADO', axis=1, inplace = True)
db.drop('HTA Y DM COMPENSADA', axis=1, inplace = True)
db.drop('Antecedentes Fliar  Enfermedad Coronaria', axis=1, inplace = True)
db.drop('CLAIFICACION IMC', axis=1, inplace = True)
db.drop('Remisiones Especialidad', axis=1, inplace = True)
db.drop('Acido Acetil Salicilico', axis=1, inplace = True)
db.drop('Proteinuria', axis=1, inplace = True)
db.drop('Adherencia al tratamiento', axis=1, inplace = True)
db.drop('inscripcion acciones', axis=1, inplace = True)
db.drop('Estadio IRC', axis=1, inplace = True)
db.drop('Complicaciones  y Lesiones en Organo Blanco', axis=1, inplace = True)
db.drop('HTA COMPENSADOS', axis=1, inplace = True)
db.drop('HTA + DM', axis=1, inplace = True)
db.drop('RCV Global', axis=1, inplace = True)
db.drop('', axis=1, inplace = True)

db.to_csv('/Users/Julian/Desktop/diabetes/db.csv')
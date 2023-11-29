import numpy as np 
import pickle 
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
#Leemos los datos con los cuales entrenaremos a la red
datos=np.loadtxt("./salida.txt")
#Utilizamos solo los datos necesarios
indice_auxiliar=6
numero_track=19
indice_final=indice_auxiliar+numero_track
idx=np.r_[0:2, indice_auxiliar:indice_final]
X=datos[:,idx]
y=datos[:,indice_final:]
datasets=train_test_split(X,y,test_size=0.2)
train_X, test_X, train_y, test_y=datasets
#Entrenamos el modelo
mlp=MLPRegressor(hidden_layer_sizes=(18,18,18),solver='adam',tol=1e-15,learning_rate='adaptive',max_iter=580000000,activation='identity')
mlp.fit(train_X,train_y)
y_pred=mlp.predict(test_X)
puntos=mlp.score(test_X,test_y)
print("\nEl modelo obtuvo los siguientes puntos: %f"%(puntos))
nombreArchivo='mlp_driver.drv'
#Guardamos la red ya entrenada
pickle.dump(mlp,open(nombreArchivo,'wb'))

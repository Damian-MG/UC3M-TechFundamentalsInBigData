# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 09:54:22 2022

@author: Pablo
"""

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing 


def show_elbowGraph(Ks,Errors):
    plt.figure(figsize=(10,5.5))
    plt.plot(Ks,Errors)
    plt.title('Elbow graph')
    plt.ylabel('average distance between points and their centroid')
    plt.xlabel('Number of clusters (k)')
    plt.show()

def avg_distance(dataset,Centroids,clasif):
    m,n = dataset.shape
    clasif = clasif.astype(int)
    avg = np.sum(np.sum((dataset[:,1:]-Centroids[clasif])**2))/m
    return avg
    
#step -1:
def get_dataset():
    #Extraemos el dataframe del csv y modificamos las variables binarias
    dataset = pd.read_csv("computers.csv")
    df_cd=pd.get_dummies(dataset["cd"])
    df_cd=df_cd.rename(columns={"yes":"cd"})
    df_laptop=pd.get_dummies(dataset["laptop"])
    df_laptop=df_laptop.rename(columns={"yes":"laptop"})
    dataset=dataset.drop(["cd"], axis=1)
    dataset=dataset.drop(["laptop"], axis=1)
    #dataset=dataset.drop(["id"], axis=1)
    dataset=pd.concat((dataset, df_cd["cd"], df_laptop["laptop"]),axis=1)
    return dataset

#step 0
def standarization(dataset):
    #estandarizamos los datos para que todas las variables tengan el mismo peso
    #la estandarización es al intervalo [0,1]
    #NO modificar el id
    min_max_scaler = preprocessing.MinMaxScaler() 
    df_escalado = min_max_scaler.fit_transform(dataset.iloc[:,1:])
    df_escalado = pd.DataFrame(df_escalado, columns=['price','speed','hd','ram','screen','cores','trend','cd','laptop'])
    data = pd.concat([dataset.iloc[:,0], df_escalado], axis=1)
    return(data)


def inicia(K, dataset):
    #Creamos los primeros centroides tomando k elementos aleatorios
    m,n = dataset.shape
    elems = random.sample(range(0,m),K)
    Centroids = np.array(dataset[elems])
    #plt.scatter(dataset["price"],dataset["ram"],c='black')
    #plt.scatter(Centroids["price"],Centroids["ram"],c='red')
    #plt.xlabel('price')
    #plt.ylabel('speed')
    #plt.show()
    return Centroids[:,1:]
#number of clusters
# Select random observation as centroids

def clasification(dataset, centroids):
    #Calculamos distancias de los puntos a los centroides 
    m,n = dataset.shape
    #creamos un array en el que guardaremos la asignación de centroide de cada elemento
    clasif= np.zeros(m)
    aux= np.zeros(len(centroids))
    for i in range(len(dataset)):
        
        for k in range(len(centroids)):
            suma = np.sum((dataset[i,1:]-centroids[k])**2)
            aux[k] = np.sqrt(suma)
            
        #asignamos a cada punto su centroide más cercano
        clase = aux.argmin()
        clasif[i] = clase
    return clasif

def update_centroids(dataset,centroids,clasif,K):
    #actualizamos centroides como la media del nuevo cluster
     newCent = np.zeros_like(centroids)
     for i in range(K):
         if i in clasif:
             newCent[i] = np.mean(dataset[np.where(clasif == i),1:], axis=1)
             
         #else:
            #newCent[i] = 0
     return newCent
        

def kmeans(dataset, K):         
    C0 = inicia(K,npdf) #centroides antiguo
    ite= 1
    clasif= clasification(npdf,C0)
    C = update_centroids(npdf,C0,clasif,K) #centroides nuevo
    error = np.sum(abs(C[:,1:]-C0[:,1:]))
    while error != 0 and ite < 100: #suele tomar entre 20 y 30 iteraciones
        ite+=1
        C0=C
        clasif= clasification(npdf,C0)
        
        C = update_centroids(npdf,C0,clasif,K)
        error = np.sum(abs(C[:,1:]-C0[:,1:]))
        #print(C)
        #print(C0)
    print('error= ' + str(error) + ' y ite= ' + str(ite))
    clasif =  clasification(npdf,C)  
    return C, clasif
        
     
            
    #data.loc[data['id'] == 4999, 'Centroids'] = 8
            
            

if __name__ == '__main__':
    
    dataset = get_dataset()
    
    data=standarization(dataset)
    npdf = data.values #TRASNFORMA EL DATAFRAME EN NP.ARRAY
    #si no os funciona esto utilizad data.to_numpy()
    #depende de la versión de pandas que tengais
    
    '''
    #Grafica SOLO de las 2 primeras variables
    plt.scatter(data["price"],data["ram"],c='black')
    plt.scatter(C[:,0:1],C[:,0:1],c='red')
    plt.xlabel('price')
    plt.ylabel('speed')
    plt.show()
    print(C)  
   '''
    distortions = []
    K = range(1,8)
    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(npdf[:,1:])
        distortions.append(kmeanModel.inertia_)
    plt.figure(figsize=(15,5.5))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()
    
    K_Dist= []
    Ks = [1,2,3,4,5,6,7]
    for k in Ks:
        C, clasif= kmeans(npdf,k)
        avgDist= avg_distance(npdf,C,clasif)
        K_Dist.append(avgDist)
        
    show_elbowGraph(Ks,K_Dist)
    kmean = KMeans(n_clusters=3).fit(npdf[:,1:])   
    print(kmean.cluster_centers_)
    C,clasif = kmeans(npdf,3)
    print(C)
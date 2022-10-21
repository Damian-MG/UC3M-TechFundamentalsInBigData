__author__      = "DAMIAN Maleno, NEREA Izcue, PABLO Alonso "
__credits__     = ["Damian Maleno", "Nerea Izcue", "Pablo Alonso"]
__version__     = "1.0"
__email__       = ["100484916@alumnos.uc3m.es","100492040@alumnos.uc3m.es","100483840@alumnos.uc3m.es"]
__status__      = "Finished"

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn import preprocessing 
import time
import multiprocessing as mp
import seaborn as sns
from matplotlib.colors import ListedColormap

MAX_K = 12
File = 'computers.csv'

'''
Method showing the Elbow Graph
'''
def showElbowGraph(Ks, errors):
    plt.figure(figsize=(10,5.5))
    plt.plot(Ks, errors)
    plt.title('Elbow graph')
    plt.ylabel('average distance between points and their centroid')
    plt.xlabel('Number of clusters (k)')
    plt.show()

'''
Method to calculate the average distance
'''
def avgDistance(dataset, centroids, classif):
    m,n = dataset.shape
    classif = classif.astype(int)
    avg = np.sum(np.sum((dataset[:,1:]-centroids[classif])**2))/m
    return avg
    
'''
Method that loads the file onto a Dataset and changes the yes/no columns by a binary 0/1
'''
def getDataset():
    dataset = pd.read_csv(File)
    df_cd = pd.get_dummies(dataset["cd"])
    df_cd = df_cd.rename(columns={"yes":"cd"})
    df_laptop = pd.get_dummies(dataset["laptop"])
    df_laptop = df_laptop.rename(columns={"yes":"laptop"})
    dataset = dataset.drop(["cd"], axis=1)
    dataset = dataset.drop(["laptop"], axis=1)
    dataset = pd.concat((dataset, df_cd["cd"], df_laptop["laptop"]),axis=1)
    return dataset

'''
Standarization method ([0,1] standarization)
'''
def standarization(dataset):
    min_max_scaler = preprocessing.MinMaxScaler() 
    df_escalado = min_max_scaler.fit_transform(dataset.iloc[:,1:])
    df_escalado = pd.DataFrame(df_escalado, columns=['price','speed','hd','ram','screen','cores','trend','cd','laptop'])
    data = pd.concat([dataset.iloc[:,0], df_escalado], axis=1)
    return (data)

'''
Method that initializes K random observations as centroids
'''
def start(K, dataset):
    m,n = dataset.shape # lines and columns # probar quitar la n
    elems = random.sample(range(0,m),K)
    centroids = np.array(dataset[elems])
    return centroids[:,1:]

'''
Method to classify associate the elements of the dataset to a centroid
'''
def classification(dataset, centroids, K):
    m,n = dataset.shape
    classif = np.zeros(m)
    aux = np.array([np.sqrt(np.sum((dataset[:,1:]-centroids[k])**2, axis = 1)) for k in range(K)])   
    classif = aux.argmin(axis= 0)
    return classif

'''
Method to update the centroid coordinates
'''
def updateCentroids(dataset, centroids, classif, K):
     new_cent = np.zeros_like(centroids)
     for i in range(K):
         if i in classif:
             new_cent[i] = np.mean(dataset[np.where(classif == i),1:], axis=1)
     return new_cent
        
'''
Method that performs the KMeans algorithm
'''
def kmeans(dataset, K):         
    i = 0
    error = 1
    prev_centroid = start(K,dataset)
    while error != 0 and i < 30:
        i += 1
        classif = classification(dataset,prev_centroid,K)
        new_centroid = updateCentroids(dataset,prev_centroid,classif,K)
        error = np.sum(abs(new_centroid-prev_centroid))
        prev_centroid = new_centroid
    return new_centroid, classif
        
'''
Method to perform X Kmeans over the same dataset and keep the best centroids
'''    
def multipleKMeans(dataframe, K, iterations=5):
    centroids = []
    list_avg = []
    for i in range(iterations):
        centroid, classif = kmeans(dataframe,K)
        avg_dist = avgDistance(dataframe,centroid,classif)
        centroids.append(centroid)
        list_avg.append(avg_dist)
    list_avg = np.array(list_avg)
    arg_min = list_avg.argmin()
    distance = min(list_avg)
    centroids = centroids[arg_min]
    return centroids, distance
    
'''
Method to perform X Kmeans over the same dataset and keep the best centroids (MP version)
'''    
def mpMultipleKMeans(dataframe, K, pool, iterdata, iterations=10):
    centroids = []
    list_avg = []
    for i in range(iterations):
        centroid, classif = mpKMeans(dataframe,K,pool,iterdata)
        avg_dist = avgDistance(dataframe,centroid,classif)
        centroids.append(centroid)
        list_avg.append(avg_dist)
    list_avg = np.array(list_avg)
    arg_min = list_avg.argmin()
    distance = min(list_avg)
    centroids = centroids[arg_min]
    return centroids, distance 

'''
Method to perform X Kmeans over the same dataset and keep the best centroids (MP version)
'''    
def mpKMeans(dataset, K, pool, iterdata):         
    i = 0
    error = 1
    prev_centroids = start(K,dataset)
    while error != 0 and i < 30:
        i += 1
        classif = mpClassification(dataset,prev_centroids,pool,iterdata)
        new_centroids = updateCentroids(dataset,prev_centroids,classif,K)
        error = np.sum(abs(new_centroids-prev_centroids))
        prev_centroids = new_centroids

    return new_centroids, classif

'''
Method to get the distance form all items to a specific centroid (MP version)
'''
def mpDistance(dataset, centroids):
    aux = []
    for centroid in centroids:
        aux.append(np.sqrt(np.sum((dataset[:,1:]-centroid)**2, axis = 1)))
    classif = np.array(aux).argmin(axis= 0)
    return(classif)

'''
Method to get the distance form all items to a specific centroid (MP version)
'''
def mpClassification(dataset, centroids, pool, iterdata):
    m,n = dataset.shape
    classif = np.zeros(m)
    new_iterdata = [(dataset[i[0]:i[1]], centroids) for i in iterdata]
    auxs = np.array(pool.starmap(mpDistance, new_iterdata), dtype=object)
    auxs = np.concatenate(auxs, axis=None)
    return auxs

'''
Method to show the heatmap
'''  
def heatmap(dataset, centroids):
    plt.figure()
    xticks_heat = dataset.columns
    xticks_heat = xticks_heat.delete(0)
    heat_data = pd.DataFrame(centroids, columns = xticks_heat)
    sns.heatmap(heat_data, cmap = "YlGnBu",yticklabels = ["Cluster 1","Cluster 2","Cluster 3","Cluster 4", "Cluster 5", "Cluster 6"])
    plt.title("Heat map of the clusters centroids")
    plt.show()

'''
Method to show the clusters in 2D
'''  
def show_clusters(dataset, classif, centroids):
    cmap_light = ListedColormap(['turquoise', 'pink', 'springgreen', 'chocolate', 'salmon', 'violet'])
    cmap_bold = ListedColormap(['lightseagreen', 'deeppink', 'limegreen', 'saddlebrown', 'orangered', 'purple'])
    plt.figure(figsize=(17,5.5))
    if classif is None:
        plt.scatter(dataset[:,1],dataset[:,2],marker='o')
    else:
        plt.scatter(dataset[:,1],dataset[:,2],c=classif,cmap=cmap_light)
    if centroids is not None:
        color = np.arange(len(centroids))
        plt.scatter(centroids[:,0],centroids[:,1],c=color,cmap=cmap_bold,marker='x',s = 400)
    plt.show()
    return    

'''
Method to chunk the dataset
'''
def chunking(Lines):
    iterdata = []
    cpus = mp.cpu_count()
    chunk_size = int(Lines/cpus)
    for i in range (cpus): iterdata.append((i*chunk_size,(i+1)*chunk_size))
    iterdata[-1] = (i*chunk_size, Lines)
    return iterdata

'''
Main program executing multiple Kmeans
'''
if __name__ == '__main__':

    # ELBOW GRAPH
    t_stamp = time.time()
    read_dataset = getDataset()
    dataframe = standarization(getDataset())
    dataset = dataframe.values
    K_dist = []
    Ks = [i for i in range(2,MAX_K+1)]
    Ks_inv = list(reversed(Ks))
    iterdata = [(dataset, K) for K in Ks_inv]
    pool = mp.Pool(mp.cpu_count())
    results = pool.starmap(multipleKMeans, iterdata)
    pool.close()
    results = np.array(results, dtype=object)
    print("Elapsed execution time of Elbow graph: ",time.time() - t_stamp," s")
    K_dist = np.flip(results[:,1])
    showElbowGraph(Ks,K_dist)

    # OPTIMAL KMEANS
    t_stamp = time.time()
    k = 6
    pool = mp.Pool(k)
    iterdata = chunking(500000)
    centroids, distance = mpMultipleKMeans(dataset,k,pool,iterdata)
    pool.close()
    print("Elapsed execution time of optimal KMeans: ",time.time() - t_stamp," s")
    
    # CLUSTER WITH HIGHER AVG_PRICE
    classif = classification(dataset,centroids,k)
    data_crudo = read_dataset.values
    avg_price =[np.mean(data_crudo[np.where(classif == i),1]) for i in range(k)]
    max_price= max(avg_price)
    max_cluster= np.array(avg_price).argmax()
    print('Cluster '+str(max_cluster+1)+' has the highest average price with: '+str(max_price))

    # HEATMAP & CLUSTERS
    classif = classification(dataset,centroids,k)
    heatmap(dataframe,centroids)
    show_clusters(dataset,classif,centroids)
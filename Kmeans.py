# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 18:54:58 2015

@author: gaoyong
"""
import os
import math
import random
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

def loadDataSet_train(path_read):
    os.chdir("data/")
    df = pd.read_csv(path_read, sep='\t', header=0, dtype=str, na_filter=False)
    return df
    
def distEclud(vecA, vecB):
    """
    欧式集合距离计算
    """
    return math.sqrt(sum(np.power((vecA-vecB).flatten().tolist(), 2)))
def distManha(vecA, vecB):
    """
    曼哈顿距离计算
    """
    return sum(np.abs(vecA-vecB))
    
def randCent(dataSet, k):
    """
    随机生成K个质心
    """
    n = dataSet.shape[1]
    centroids = np.mat(np.zeros([k,n]))
    for j in range(n):
        minJ = min(dataSet[:,j])
        maxJ = max(dataSet[:,j])
        rangeJ = float(maxJ - minJ)
        centroids[:,j] = minJ + rangeJ*np.random.rand(k,1)
    return centroids 
    
def selectCent(dataSet, k):
    """
    从样本点中随机生成K个质心
    """
    num_rows = dataSet.shape[0]
    #在范围A～B内随机生成n个不重复随机数random.sample(range(A,B+1),n)
    list_k = sorted(random.sample(range(0,num_rows),k))
    centroids = np.mat(dataSet[list_k])
    return centroids
      
def Kmeans(dataSet, k_clusters, distMeas =distEclud, createCent =selectCent):
    dataSet = np.asarray(dataSet)
    num_rows = dataSet.shape[0]
    clusterAssment = np.matrix(np.zeros([num_rows, 2]))
    centroids = createCent(dataSet, k_clusters)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(num_rows):
            minDist = np.inf
            minIndex = -1
            for j  in range(k_clusters):
                dist_ij = distMeas(dataSet[i], np.array(centroids[j]))
                if dist_ij <minDist:
                    minDist = dist_ij; minIndex = j
            if clusterAssment[i, 0] != minIndex: clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist
        if not clusterChanged:
            break
        #更新质心 
        for i in range(k_clusters):
            ptsInClust= dataSet[np.nonzero(clusterAssment[:,0].A == i)[0]]
            centroids[i,:] = np.mean(ptsInClust, axis = 0)
    return centroids, clusterAssment
def calSC(dataSet, clusterAssment, k_clusters, distMeas =distEclud):
    """
    计算轮廓系数
    """
    S = []
    dataSet = np.asarray(dataSet)
    for i in range(k_clusters):        
        cluster = dataSet[np.nonzero(clusterAssment[:,0].A == i)[0]]
        cluster_other = dataSet[np.nonzero(clusterAssment[:,0].A != i)[0]]

        for j in range(cluster.shape[0]):
            minDistOtherCluster = np.inf
            aveDistInCluster = 0
            for k in range(cluster.shape[0]):
                aveDistInCluster = aveDistInCluster + distMeas(cluster[j], cluster[k])
            aveDistInCluster = aveDistInCluster/float(cluster.shape[0])
            for k in range(cluster_other.shape[0]):
                dist_jk = distMeas(cluster[j], cluster_other[k])
            if cluster_other.shape[0]!=0:
                if dist_jk <minDistOtherCluster:
                    minDistOtherCluster = dist_jk
                    s = (minDistOtherCluster - aveDistInCluster)/max(minDistOtherCluster, aveDistInCluster)
            else:
                s = -1
            S.append(s)
    return S, sum(S)/len(S)
            
        
if __name__ == "__main__":
    path_read = "cluster.txt"
    k_clusters=10
    df = loadDataSet_train(path_read)
    SSE = []
    SC = []
    dataSet = np.matrix(df).astype(np.float)
    for i in range(2, k_clusters+1):
        centroids, clusterAssment = Kmeans(dataSet, i)
        sse = np.sum(clusterAssment[:,1])
        SSE.append(sse)
        S, sc = calSC(dataSet, clusterAssment, i)
        SC.append(sc)
    plt.plot(range(2, k_clusters+1), SC, 'r*')
    plt.plot(range(2, k_clusters+1), SC, 'b-')
#    plt.plot(range(2, k_clusters+1), SC, 'r*')
#    plt.plot(range(2, k_clusters+1), SC, 'b-')

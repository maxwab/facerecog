#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 16:31:57 2017

@author: maxwab
"""

import numpy as np
from PIL import Image
import os, sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from features_reduction import *
import time
import random
from features_reduction import *

def delta(x,i,classs):
    '''
    fonction indicatrice de la classe i
    '''
    n,m = len(x),len(classs)
    
    if (n != m):
        print 'vectors of differents sizes, cannot operate delta'
        
    tmp = i*np.ones(n)-classs

    for k in range(n):
        if tmp[k]==0:
            tmp[k]=1
        else:
            tmp[k]=0 
            
    return tmp*x


# Définition de la fonction de résidu qui renvoie la classe minimisant l'erreur de reconstruction suivant la norme L2.

def residu(y,A,x,class_x):
    '''
    renvoie les residus pour chaque classe.
    '''
    k = np.max(class_x)+1
    r = np.zeros(k)
    
    for i in range(0,k):
        r[i] = np.linalg.norm(y - np.dot(A,delta(x,i,class_x)))
        
    return r


# Définition de la fonction retournant l'indice de concentration (_Sparsity Concentration Index_)


def SCI(x,classs):
    '''
    @input
      - classs: classe de chaque training element.
      - x     : sparse coefficients
    '''
    
    k = len(set(classs)) # Nombre de classes différentes
    
    # Puis on retourne la valeur du SCI
    return (k*(1/np.linalg.norm(x,ord=1))*np.max([np.linalg.norm(delta(x,i,classs),ord=1) for i in range(k)]) - 1)/(k-1)
    


# **Utilitaire** : trouver la classe d'un élément du test set pour le Yale Database


def find_class(i):
    return int(i)/12

def noise_image(image_input, per=0.5):
    '''
    Ajoute du bruit aléatoire à une image.
    @params:
        per : pourcentage de pixels à corrompre
    '''
    sz0 = image_input.shape[0]
    sz1 = image_input.shape[1]
    
    # Création du masque
    nb_pix_to_noise = int(np.floor(per*sz0*sz1))
    print nb_pix_to_noise
    
    mask = np.ones((sz0*sz1,1))
    ids = np.random.permutation(sz0*sz1)[0:nb_pix_to_noise]
    mask[ids] = 0
    mask = np.reshape(mask,(sz0,sz1))
    
    # Matrice de pixels aléatoires (intensité entre 0 et 255)
    rand_pix = np.random.randint(0,256,size=(sz0,sz1))
    
    # On retourne la matrice initiale où l'on a changé les pixels indiqués par des pixels aléatoires
    return np.multiply(mask,image_input) + np.multiply(1-mask,rand_pix)
    
def black_frame(array_orig, x0, x1, y0, y1):
    '''
    Ajoute un bandeau noir dont les coins sont de coordonnées (x0,y0),(x1,y1),(x0,y1) et (x1,y0)
    '''
    array_tmp = np.ones_like(array_orig)
    
    if ((x0 > array_tmp.shape[1]) | (x1 > array_tmp.shape[1]) | (y0 > array_tmp.shape[0]) | (y1 > array_tmp.shape[0])):
        print "Error : coordonnees du bandeau trop grandes pour l'image"
    
    for i in range(array_tmp.shape[1]):
        for j in range(array_tmp.shape[0]):
            if ((i >= min(x0,x1)) & (i <= max(x0,x1)) & (j >= min(y0,y1)) & (j <= max(y0,y1))):
                array_tmp[j][i] = 0
    
    return array_tmp*array_orig
    
def matrix_transform(X):
    '''
    Ici X est une liste de np.array (sous forme de matrice, pas de vecteur)
    '''
    
    X_toconcat = [np.reshape(e,(X[0].shape[0]*X[0].shape[1],1)) for e in X]
    
    # Puis concaténation pour avoir une matrice unique
    return np.concatenate(X_toconcat,axis=1) # Liste des samples du train, concaténés en colonne.
    
    
###### Algorithme
    

def SRC(Xtrain, Xtest, ytrain, type_feature_reduc=None, reduce_lines=12, reduce_columns=10, lambda_val=0.02, per_bruit=0.3, pos_occl=None):
    '''
    @params :
        * Xtrain : iterable of numpy arrays representing faces
        * Xtest : iterable of numpy arrays representing faces
        * per_bruit : pourcentage de pixels à remplacer par du bruit uniforme sur [0,255]
        * pos_occl : position du rectangle pour occlure (xgauche,yhaut,xdroite,ybas)
    '''
    
    # ---- Définissons les paramètres
    
    n_train = len(Xtrain)
    n_test = len(Xtest)
    
    n_components = reduce_lines*reduce_columns
    
    k = np.max(ytrain)+1 #Nb de classes
    
    
    s0 = Xtrain[0].shape[0]
    s1 = Xtrain[0].shape[1]



    # ---- Dans un premier temps on bruite l'ensemble de test si nécessaire
    
    if (per_bruit != None):
        tmp = np.copy(Xtest)
        Xtest = [noise_image(e,per=per_bruit) for e in tmp]

    # ---- Ensuite on corrompt l'ensemble de test si nécessaire avec une occlusion
    
    if (pos_occl != None):
        xgauche,yhaut,xdroite,ybas = pos_occl[0],pos_occl[1],pos_occl[2],pos_occl[3]
        tmp = np.copy(Xtest)
        Xtest = [black_frame(e,xgauche,xdroite,ybas,yhaut) for e in tmp]
                 
    # ---- Transformation en deux matrices plutôt que deux listes de matrices
    
    Xtrain = matrix_transform(Xtrain)
    Xtest = matrix_transform(Xtest)
                 
    # ---- Normalisation
    
    ss = StandardScaler()
    # Note : on normalise les deux séparéments car on est juste en train de ramener chaque photo à une longueur unité suivant la norme 2
    Xtrain = ss.fit_transform(Xtrain)
    Xtest = ss.fit_transform(Xtest)
        
    # ---- Ensuite on fait une réduction de dimension pour les deux
    
    # Plusieurs cas :
    # - None (classic) : on resize juste avec nearest
    # - fisherfaces
    # - randomfaces
    # - eigenfaces
    
    
    # Note : on n'oublie pas de transposer pour pouvoir utiliser les fonctions de features_reduction

    if (type_feature_reduc == 'eigenfaces'):
        Xtrain, Xtest = eigenfaces(Xtrain.T,Xtest.T,n_components=n_components)
        Xtrain, Xtest = Xtrain.T, Xtest.T
    elif (type_feature_reduc == 'fisherfaces'):
        Xtrain, Xtest = fisherfaces(Xtrain.T,ytrain,Xtest.T,n_components=n_components)
        Xtrain, Xtest = Xtrain.T, Xtest.T
    elif (type_feature_reduc == 'randomfaces'):
        Xtrain, Xtest = randomfaces(Xtrain.T, Xtest.T, n_components=n_components)
        Xtrain, Xtest = Xtrain.T, Xtest.T
    else: # Cas classic
        # On est obligé de réutiliser PIL ici ... donc de reshape à nouvea chaque colonne et d'avoir une liste d'éléments !
        List_Xtrain, List_Xtest = [], []
    
        for j in range(Xtrain.shape[1]):
            tmp = np.reshape(Xtrain[:,j],(s0,s1))
            im = Image.fromarray(tmp)
            im = im.resize((reduce_lines,reduce_columns), Image.NEAREST) # Valeurs indiquées en paramètre de la fonction  
            List_Xtrain.append(np.asarray(im, dtype=np.float64))
        Xtrain = matrix_transform(List_Xtrain)
            
        for j in range(Xtest.shape[1]):
            tmp = np.reshape(Xtest[:,j],(s0,s1))
            im = Image.fromarray(tmp)
            im = im.resize((reduce_lines,reduce_columns), Image.NEAREST) # Valeurs indiquées en paramètre de la fonction  
            List_Xtest.append(np.asarray(im, dtype=np.float64))
        Xtest = matrix_transform(List_Xtest)
    
        
        
    # ---- Ensuite on applique la minimisation Lasso pour chaque exemple du test set
    
    # Rappel : 
    # * y : Elément à tester (est une colonne de Xtest)
    # * Xtrain : Matrice A des exemples d'entrainement
    # * x : coefficients issus de la minimisation LASSO
    # * ytrain : classe des exemples d'entrainement

    
    preds = np.zeros(Xtest.shape[1])
    rejections = np.zeros(Xtest.shape[1])
    residus = np.zeros((k,Xtest.shape[1]))


    # on crée un classifieur Lasso avec le lambda spécifié en paramètre
    clf = Lasso(alpha=lambda_val) 
    
    # Pour chaque exemple à tester on génère les coefficients et on prend la meilleure prédiction
    for j in range(Xtest.shape[1]):
        y = Xtest[:,j] # Exemple courant à tester
        clf.fit(Xtrain,y)
        x = clf.coef_

        # On fait la prédiction
        residus[:,j] = residu(y,Xtrain,x,ytrain)
        preds[j] = np.argmin(residu(y,Xtrain,x,ytrain)) 
        rejections[j] = SCI(x,ytrain)
    
    
    # ---- On renvoie les valeurs ci-dessus
    
    # predictions : vecteur des prédictions pour chaque exemple du test set
    # sci : vecteur des SCI pour chaque élément du test set
    # residus : matrice des résidus : m lignes (nombre de classes), n colonnes (nombre d'éléments du test set)
    return preds, rejections, residus
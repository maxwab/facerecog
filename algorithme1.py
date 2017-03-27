#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 16:31:57 2017

@author: maxwab
"""

def simulation(Xtrain_raw, Xtest_raw, ytrain, ytest, lambda_val=0.02, type_feature_reduc='classic', per_bruit=0.3, pos_occl=(c1,c2,c3,c4)):
    
    # Dans un premier temps on corrompt l'ensemble de test si nécessaire
    
    # Ensuite on 
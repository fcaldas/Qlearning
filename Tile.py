# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 15:59:34 2016

@author: fcaldas

Class used for coarse tilling: implements in disk storage
"""
import numpy as np
import shelve
import random
import string

class Tiling:
    
    def __init__(self, 
                 minV=np.array([-0.2, -5, -pi, -pi*4]),
                 maxV=np.array([0.2, 5, pi, pi*4]),
                 ntiles=np.array([40.,40.,360.,200.]),
                 nOutputs=20):
        self.Q = {}
        self.inflim = minV
        self.suplim = maxV
        self.nOutputs = nOutputs
        fname = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
        
        self.bin0 = np.linspace(self.inflim[0], self.suplim[0], num=self.ntiles[0])
        self.bin1 = np.linspace(self.inflim[1], self.suplim[1], num=self.ntiles[1])
        self.bin2 = np.linspace(self.inflim[2], self.suplim[2], num=self.ntiles[2])
        self.bin3 = np.linspace(self.inflim[3], self.suplim[3], num=self.ntiles[3])
        
        self.Q = shelve.open()
        
    def getIndex(self, v):
        #limit v3 to 2piXl[2] = X[2] % 2*pi;
        pos = v
        if(pos[2] > pi):
            pos[2] = 2*pi - pos[2];
        elif(pos[2] < -pi):
            pos[2] = 2*pi + pos[2];
        
        pos[0] = np.min([pos[0], self.maxV[0]])
        pos[1] = np.min([pos[1], self.maxV[1]])
        pos[2] = np.min([pos[2], self.maxV[2]])
        pos[3] = np.min([pos[3], self.maxV[3]])
        idx = [0,0,0,0]
        idx[0] = np.argmax(pos[0] <= self.bin0 )
        idx[1] = np.argmax(pos[1] <= self.bin1 )
        idx[2] = np.argmax(pos[2] <= self.bin2 )
        idx[3] = np.argmax(pos[3] <= self.bin3 )
        return idx
    
    def getValuesAtIndex(i):
        pos = str(i[0]) + "-" + str(i[1]) + "-" + str(i[2]) + "-" + str(i[3])
        vs = np.zeros(self.nOutputs)
        for i in range(0, nOutputs):
            if(pos + "-" + str(i) in self.Q):
                vs[i] = self.Q[pos + "-" + str(i)]
        return vs
            
    def greedy(self, v):
        i = self.getIndex(v)
        candidates = self.getValuesAtIndex(i)
        iargmax = np.random.choice(np.where(candidates == np.max(candidates))[0])
        return iargmax
        
    def get(s, a):
        i = getIndex(s)
        pos = str(i[0]) + "-" + str(i[1]) + "-" + str(i[2]) + "-" + str(i[3]) + "-" + str(a)
        if(pos in self.Q):
            return self.Q[pos]
        else:
            return 0
    
    def set(s, a, v):
        pos = str(i[0]) + "-" + str(i[1]) + "-" + str(i[2]) + "-" + str(i[3]) + "-" + str(a)
        self.Q[pos] = v
        
        
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 21:56:02 2016

@author: fcaldas
"""

from pendulum import *
from Qlearning import *
import matplotlib.pyplot as plt
import seaborn
import numpy as np
import tqdm


def runSimGetStates(sim, learner, S, maxSteps = 5000):
    stateList = np.zeros([maxSteps, 4])
    actions = np.zeros(maxSteps)
    running = True
    sim.restart(S)
    action = learner.setInitialState(S)
    sim.step(action * 12)
    iStep = 0
    while(running and iStep < maxSteps):
        stateList[iStep, :] = sim.getState()
        r = sim.getReward()
        action =learner.getAction(sim.getState(), r)
        sim.step(action * 12)
        actions[iStep] = action * 12
        print(learner.getStateIndex(sim.getState()))
        if(sim.stopSim()):
            running = False;
        iStep += 1
    stateList[:, 2] = (stateList[:, 2]+pi) % (2*pi)
    plt.figure()    
    plt.title("Phi and X")
    plt.plot(stateList[:iStep, 0])
    plt.plot(stateList[:iStep, 2])
    plt.show()
    plt.figure()
    plt.plot(actions[:iStep])
    plt.show()
    return stateList[:iStep, :]
    

def runSim(sim, learner, S, maxSteps = 10000):
    running = True
    sim.restart(S)
    action = learner.setInitialState(S)
    sim.step(action * 12)
    iStep = 0
    while(running and iStep < maxSteps):
        r = sim.getReward()
        action =learner.getAction(sim.getState(), r)
        sim.step(action * 12)
        if(sim.stopSim()):
            running = False;
        iStep += 1
    return iStep
        

sim = Simulator()
sim.restart([0., 0., 0., 0.])
learner = Qlearner() #pickle.load( open( "save.p", "rb" ) )
learner.epsilon = 0.9
for k in range(0,40):
    
    learner.epsilon_decay_rate = 0.9999
    iMax = np.zeros(10000)
    for i in tqdm.tqdm(range(0,10000)):
        iMax[i] = runSim(sim, learner, [0, 0, np.random.rand() * 0.05 - 0.05 , 0.])
fd = open("bestSoFar10S.obj", "wb")
pickle.dump(learner, open("bestSoFar.obj", "wb"))
fd.close()
plt.plot(iMax)
plt.show()
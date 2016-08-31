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
from SARSA import *

def runSimNoLearner(sim, S, maxSteps=500): # 5 seconds
    stateList = np.zeros([maxSteps, 4])
    sim.restart(S)
    iStep = 0
    while(iStep < maxSteps):
        stateList[iStep, :] = sim.getState()
        sim.step(0.)
        iStep += 1
    plt.figure()    
    plt.title("X")
    plt.plot(stateList[:iStep, 0])
    plt.show()
    plt.figure()
    plt.title("Phi")
    plt.plot(stateList[:iStep, 2])
    plt.show()
    return stateList[:iStep, :]
    


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
  #  stateList[:, 2] = (stateList[:, 2]+pi) % (2*pi)
    plt.figure()    
    plt.title("Phi and X")
    plt.plot(stateList[:iStep, 0], label="x")
    plt.plot(stateList[:iStep, 2], label="o")
    plt.legend()
    plt.show()
    plt.figure()
    plt.plot(actions[:iStep])
    plt.show()
    plt.figure()
    plt.title('Speeds')
    plt.plot(stateList[:iStep, 1], label="x'")
    plt.plot(stateList[:iStep, 3], label="o'")
    plt.legend()
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
        

sim = Simulator(dt = 0.01)
sim.restart([0., 0., 0., 0.])
learner = SARSAlearner() 
#learner.Q = np.load("bestQ.obj")
iRecord = 0
learner.alpha = 0.4
learner.epsilon = 0.05
for k in range(0,300):
    #learner.epsilon = 0.05
    learner.epsilon_decay_rate = 0.9999
    iMax = np.zeros(20000)
    for i in tqdm.tqdm(range(0,20000)):
        iMax[i] = runSim(sim, learner, [0, 0, np.random.rand() * 0.1 * 2 - 0.1 , 0.])
        if(iMax[i] > iRecord):
            iRecord = iMax[i]
            print "\nNew record : ", iRecord, " steps"
    print "Avg steps = ", np.mean(iMax), '  -  ', np.std(iMax)

np.save("bestQ.obj", learner.Q)
plt.plot(iMax)
plt.show()
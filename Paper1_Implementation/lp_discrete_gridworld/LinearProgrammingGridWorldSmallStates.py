#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import random
import copy
from GridWorld import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# In[2]:


# Directions
#0 is left
#1 is down
#2 is right
#3 is Up


# In[3]:


sampleGrid=myGridWorld()
sampleGridTrainer=myGridWorldTrainer()
sampleTrajectories=sampleGridTrainer.allInOne(sampleGrid,20)
for i in sampleTrajectories:
    print(i)
    
for i in sampleGridTrainer.matrix:
    print(i)
    
for i in sampleGridTrainer.DirectionalMatrix:
    print(i)
    
for i in sampleGridTrainer.Q:
    print(i)
    


# In[82]:


plt.matshow(sampleGridTrainer.env.RewardGrid);
plt.colorbar()
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = y = np.arange(0, 5, 1)
X, Y = np.meshgrid(x, y)
zs = sampleGridTrainer.env.RewardGrid
Z = zs.reshape(X.shape)
ax.view_init(45, -135)
ax.plot_surface(X, Y, Z,alpha=0.5,cmap='jet', rstride=1, cstride=1, edgecolors='k', lw=1)
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Reward Values')

plt.show()


# In[5]:


def constructTransitionMatrix(size=5,states=25,actions=4,noisyMoveChance=0.3):
    
    
    transitionMatrix=np.zeros([states,states,actions])
    
    for state in range(states):
        for action in range(actions):
            i=int(state/size)
            j=state%size
            
            probStay=1
            
            if(action==0):#Left
                if(0<j):
                    probStay=probStay-(1-noisyMoveChance)
                    j2=j-1
                    transitionMatrix[state][int(i*size+j2)][action]=1-noisyMoveChance
            elif(action==1):#1 is down
                if(i<size-1):
                    probStay=probStay-(1-noisyMoveChance)
                    i2=i+1
                    transitionMatrix[state][int(i2*size+j)][action]=1-noisyMoveChance
            elif(action==2):#2 is right
                if(j<size-1):
                    probStay=probStay-(1-noisyMoveChance)
                    j2=j+1
                    transitionMatrix[state][int(i*size+j2)][action]=1-noisyMoveChance
            elif(action==3):#3 is Up
                if(0<i):
                    probStay=probStay-(1-noisyMoveChance)
                    i2=i-1
                    transitionMatrix[state][int(i2*size+j)][action]=1-noisyMoveChance
                    
            
            if(0<j):
                probStay=probStay-(noisyMoveChance/4)
                j2=j-1
                transitionMatrix[state][int(i*size+j2)][action]+=(noisyMoveChance/4)
            if(i<size-1):
                probStay=probStay-(noisyMoveChance/4)
                i2=i+1
                transitionMatrix[state][int(i2*size+j)][action]+=(noisyMoveChance/4)
            if(j<size-1):
                probStay=probStay-(noisyMoveChance/4)
                j2=j+1
                transitionMatrix[state][int(i*size+j2)][action]+=(noisyMoveChance/4)
            if(0<i):
                probStay=probStay-(noisyMoveChance/4)
                i2=i-1
                transitionMatrix[state][int(i2*size+j)][action]+=(noisyMoveChance/4)
            
            if(probStay<10**-15):
                probStay=0
            transitionMatrix[state][state][action]=probStay
    return transitionMatrix           
        


# In[6]:


transitionMatrix=constructTransitionMatrix()


# In[7]:


optimumPolicy=[[2,2,2,2,2],[3,2,2,3,3],[3,3,3,3,3],[3,3,2,3,3],[3,2,2,2,3]]
DirectionalMatrix=[]
for i in range(5):
            row=[]
            for j in range(5):
                if(optimumPolicy[i][j]==0):
                    row.append('\u2190')
                elif(optimumPolicy[i][j]==1):
                    row.append('\u2193')
                elif(optimumPolicy[i][j]==2):
                    row.append('\u2192')
                elif(optimumPolicy[i][j]==3):
                    row.append('\u2191')
            DirectionalMatrix.append(row)
for i in DirectionalMatrix:
    print(i)


# In[8]:


import numpy
from scipy.optimize import linprog

def performInverseReinforcementLearning( policy, gamma=0.5, l1=10):
    trans_probs=constructTransitionMatrix()
    conditions=[]
    c = np.zeros([3 * 25])
    
    for i in range(25):
        optimalAction = policy[i]
        tempTransProbMatrix= gamma * trans_probs[:, :, optimalAction]
        tempInverse = np.linalg.inv(np.identity(25) - tempTransProbMatrix)
        
        for j in range(4):
            if(j != optimalAction):
                condition= - np.dot(trans_probs[i, :, optimalAction] - trans_probs[i, :, j], tempInverse)
                conditions.append(condition)
    
    equality=np.zeros(625)
    for i in range(25):
        c[25:2 * 25] = -1
        c[2 * 25:] = l1
    conditions=np.array(conditions)
    conditions=np.reshape(conditions,[625,75])
    print(len(c),conditions.shape)
    rewards=linprog(c, A_ub=conditions, b_ub=equality)
    
    #rewards = rewards/max(rewards)
    return rewards


# In[9]:


#policy=np.reshape(sampleGridTrainer.matrix,[25,1])
policy=np.reshape(optimumPolicy,[25,1])
reward=performInverseReinforcementLearning(policy,0.5,10)
reward=reward['x'][:25]
reward=np.reshape(reward,[5,5])
plt.matshow(reward);
plt.colorbar()
plt.show()


# In[10]:


import numpy
from cvxopt import matrix, solvers
from scipy.optimize import linprog

def initializeSolverMatrix(penalty=10):
    #For all states and all possible non-optimal actions to all states
    A = np.zeros([25**2, 3 * 25])
    b = np.zeros([25**2])
    x = np.zeros([3 * 25])
    
    def initialize():
        size=25
        num=150
        i=0
        while(i<25):
            
            A[num+ i, i] = 1
            A[num + size + i, i] = -1
            j=2
            while(j<4):
                A[num + j * size + i, i] = 1
                A[num + j * size + i, 2 * size + i] = -1
                j+=1
            b[num + i] = 1
            b[num + size + i] = 0
            i+=1
    
    i=0
    initialize()
    while(i<25):
        x[25:] = -1
        x[-25:] = penalty
        i+=1
    return A,b,x

def optimisedIRL( policy, gamma=0.5, penalty=10):
    
    TransitionMatrix=constructTransitionMatrix()
    
    A,b,x=initializeSolverMatrix(penalty)
    i=0
    while (i<25):
        optimalAction = int(policy[i])
        tempTransProbMatrix= gamma * TransitionMatrix[:, :, optimalAction]
        patialInvertedMatrix = np.linalg.inv(np.identity(25) - tempTransProbMatrix)

        temp = 0
        j=0
        while(j<4):
            if j != optimalAction:
                otherPartialMatrix=TransitionMatrix[i, :, optimalAction] - TransitionMatrix[i, :, j]
                val=- np.dot(otherPartialMatrix, patialInvertedMatrix)
                pos=25*3
                A[i * 3 + temp, :25] = val
                A[pos+ i *3+temp,:25] = val
                A[pos+ i *3+temp, 25+i]=1
            else:
                temp=temp-1
            temp =temp+1
            j=j+1
        i=i+1
    x=matrix(x)
    A=matrix(A)
    b=matrix(b)
    return solvers.lp(x,A,b)


# In[99]:


error=[]
index=[]
#for i in range(0,10):
for i in range(1):
    #policy=np.reshape(sampleGridTrainer.matrix,[25,1])
    policy=np.reshape(optimumPolicy,[25,1])
    rewards=optimisedIRL(policy,0.1,3)
    rewards = rewards['x']
    rewards=rewards[:5*5]
    rewards = rewards/max(rewards)
    rewards=np.reshape(rewards,[5,5])
    true=np.abs(sampleGridTrainer.env.RewardGrid)
    obtained=np.abs(rewards)
    errors=true-obtained
    #print(errors)
    errors=np.abs(sum(sum(errors)))
    error.append(errors)
    index.append(i)
    plt.matshow(rewards);
    plt.colorbar()
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(0, 5, 1)
    X, Y = np.meshgrid(x, y)
    zs = rewards
    Z = zs.reshape(X.shape)
    ax.view_init(45, 135)
    ax.plot_surface(X, Y, Z,alpha=0.5,cmap='jet', rstride=1, cstride=1, edgecolors='k', lw=1)

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Reward Values')

    plt.show()


# In[ ]:





# In[108]:


#conda install cvxopt


# In[43]:



b=[(-1,1) for i in range(5*5)]


# In[44]:





# In[ ]:





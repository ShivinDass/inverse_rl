import gym
import numpy as np
import random,copy
import pickle
from math import exp,pi,sqrt


def getState(s):
	xpos = min(discretization-1,round((s[0]-xmin)/x_binsize))
	vpos = min(discretization-1,round((s[1]-vmin)/v_binsize))
	return int(xpos),int(vpos)

def calcPDF(a,m,s):
	a=(a-m)/s
	return (exp(-(a)**2/(2))/(sqrt(2*pi)))/s



def getFunction(Q,mean,scale,i):
	alpha=0.1
	gamma=0.99
	epsilon=0.05
	V=[[0 for v in range(120)] for x in range(120)]
	
	time=0
	while time<10000:# or min_steps>260:
		obs = env.reset()

		while 1>0:
			x,v = getState(obs)
			#a=random.randint(0,2)
			#if random.random()<1-epsilon:
				#print(Q[x][v])
			a=np.argmax(np.array(Q[x][v]))

			obs,R,done,info=env.step(a)
			R = calcPDF(x,mean,scale)

			x1,v1 = getState(obs)			

			V[x][v] += alpha*(R + gamma*V[x1][v1] - V[x][v])
			if done:
				break
		
		time+=1
		print("episode:",time)
	print(V)

	# f = open('V'+str(i),'wb')
	# pickle.dump(V,f)
	# f.close()





env=gym.make('MountainCar-v0')
#env.seed(1)

e=env.getenv()
#print(env._max_episode_seconds,env._max_episode_steps)
discretization=120
xmin = e.min_position
xmax = e.max_position
x_binsize=(xmax - xmin)/discretization
#print(xmin,xmax,x_binsize)

vmin = -1*e.max_speed
vmax = e.max_speed
v_binsize=(vmax - vmin)/discretization
#print(vmin,vmax,v_binsize)

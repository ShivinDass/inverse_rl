import gym
import numpy as np
import random,copy
import pickle


def getState(s):
	xpos = min(discretization-1,round((s[0]-xmin)/x_binsize))
	vpos = min(discretization-1,round((s[1]-vmin)/v_binsize))
	return int(xpos),int(vpos)

env=gym.make('MountainCar-v0')
#env.seed(1)

alpha=0.1
gamma=0.99
epsilon=0.05
Q=[[[0 for a in range(3)] for v in range(120)] for x in range(120)] 
Q_optim=copy.deepcopy(Q)
min_steps=env._max_episode_steps+1

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

time=0
while time<20000:# or min_steps>260:
	obs = env.reset()
	score=0
	steps=0

	greedPol=random.randint(0,9)
	while 1>0:
		steps+=1
		if time%2000==0:
			env.render()
		#print(env.state)
		x,v = getState(obs)
		a=random.randint(0,2)
		if greedPol==0 or random.random()<1-epsilon:
			#print(Q[x][v])
			a=np.argmax(np.array(Q[x][v]))

		obs,R,done,info=env.step(a)
		
		x1,v1 = getState(obs)

		score += R
		#if x1>0.2:
		#	R+=obs[0]


		Q[x][v][a] += alpha*(R + gamma*max(Q[x1][v1]) - Q[x][v][a]) 
		

		if done:
			break
	
	time+=1
	print("episode:",time,steps)

	if greedPol==0 and steps<min_steps:
		min_steps=steps
		Q_optim=copy.deepcopy(Q)

# f = open('Q_Opt1','wb')
# pickle.dump(Q,f)
# f.close()

while input()=="Y":
	obs=env.reset(-0.5)
	while 1>0:
		
		env.render()
		
		x,v = getState(obs)
		#print(obs,x,v)
		a=np.argmax(np.array(Q[x][v]))

		obs,R,done,info=env.step(a)
		
		if done:
			break
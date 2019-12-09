import pickle,random
import calcValueFunc
import numpy as np
import gym
from scipy.optimize import linprog
from math import exp,pi,sqrt
import matplotlib.pyplot as plt


def calcPDF(a,m,s):
	a=(a-m)/s
	return (exp(-(a)**2/(2))/(sqrt(2*pi)))/s

def getSampleStates(sampleTrajs):
	sample_states_x=[]
	sample_states_v=[]
	l=[i for i in range(113)]
	for i in range(sampleTrajs):#50
		k=random.choice(l)
		l.remove(k)
		sample_states_x.append(k)
	l=[i for i in range(120)]
	for i in range(sampleTrajs):#50
		k=random.choice(l)
		l.remove(k)
		sample_states_v.append(k)
	sample_states_x.sort()
	sample_states_v.sort()

	S=[]
	for i in sample_states_x:
		for j in sample_states_v:
			S.append([i,j])
	return S


def getState(s):
	xpos = min(discretization-1,round((s[0]-xmin)/x_binsize))
	vpos = min(discretization-1,round((s[1]-vmin)/v_binsize))
	return [int(xpos),int(vpos)]

def getNextState(s,a):
	x=x_binsize*s[0]+xmin
	v=v_binsize*s[1]+vmin
	#print(x,v)
	env.reset(x,v)
	obs,R,done,info=env.step(a)

	return getState(obs)


random.seed(0)
env=gym.make('MountainCar-v0')
original=False
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



f = open('Data\\Q_Opt1','rb')
Q=pickle.load(f)
f.close()

#print(Q)
numOfBasis=26
scale=0.5
basis_bin=(0.5+1.2)/(numOfBasis-1)
basis=[-1.2+basis_bin*i for i in range(numOfBasis-1)]+[0.5]
#print(basis)
#value_funcs=[]


#for i in range(1,numOfBasis):
#	calcValueFunc.getFunction(Q,basis[i],scale,i)


V=[]
for i in range(numOfBasis):
	f = open('Data\\V'+str(i),'rb')
	V.append(pickle.load(f))
	f.close()
#print(len(value_funcs[0][0]))

kk=50
if original:
	kk=10
S0=getSampleStates(kk)
print(len(S0))

alphas=None
if original:
	c = [0]*numOfBasis + [-1]*len(S0)
	bound=[(-1,1) for i in range(numOfBasis)] + [(None,None) for j in range(len(S0))]
	A = [[0 for i in range(len(c))] for j in range(4*len(S0))] #list(np.zeros((len(S0)*2,len(c))))
	b =	[0 for j in range(4*len(S0))]
	#print(A)
	#print(len(A),len(A[0]))

	for i in range(len(S0)):
		A[4*i][numOfBasis+i]=1
		A[4*i+1][numOfBasis+i]=1
		A[4*i+2][numOfBasis+i]=1
		A[4*i+3][numOfBasis+i]=1

		a=np.argmax(np.array(Q[S0[i][0]][S0[i][1]]))
		actions=[0,1,2]
		actions.remove(a)
		a2=actions[0]
		a3=actions[1]

		s=getNextState(S0[i],a)
		s2=getNextState(S0[i],a2)
		s3=getNextState(S0[i],a3)
		
		fact=3
		for j in range(numOfBasis):
			A[4*i][j] = (-1*V[j][s[0]][s[1]] + V[j][s2[0]][s2[1]])/10000
			A[4*i+1][j] = fact*(-1*V[j][s[0]][s[1]] + V[j][s2[0]][s2[1]])/10000
			A[4*i+2][j] = (-1*V[j][s[0]][s[1]] + V[j][s3[0]][s3[1]])/10000
			A[4*i+3][j] = fact*(-1*V[j][s[0]][s[1]] + V[j][s3[0]][s3[1]])/10000


		#print(s,s2,s3)

	#print(A)
	print("Solving LP")
	res=linprog(c,A_ub=A,b_ub=b,bounds=bound)
	# f=open('result','wb')
	# pickle.dump(res,f)
	# f.close()
	print(res)
	alphas=list(res['x'])
	alphas=alphas[:numOfBasis]
	print(alphas)
	#exit(0)
else:
	c = [0]*(numOfBasis)
	bound=[(-1,1) for i in range(numOfBasis)]

	for i in range(len(S0)):

		a=np.argmax(np.array(Q[S0[i][0]][S0[i][1]]))
		actions=[0,1,2]
		actions.remove(a)
		r=random.randint(0,1)
		a2=actions[r]

		s=getNextState(S0[i],a)
		s2=getNextState(S0[i],a2)
		
		for j in range(numOfBasis):
			tmpV= -1*V[j][s[0]][s[1]] + V[j][s2[0]][s2[1]]
			if tmpV>0:
				tmpV*=1.85 #1.85
			c[j] += tmpV
			
		#print(s,s2,s3)

	print(c)
	print("Solving LP")
	res=linprog(c,bounds=bound)
	print(res)
	alphas=res['x']

X=[]
R=[]
x=xmin
std=0.15
if original:
	std=0.5
while x<=xmax:
	X.append(x)
	rx=0
	for k in range(len(basis)):
		rx+=alphas[k]*calcPDF(x,basis[k],std) #0.15,0.5
	R.append(rx)

	x+=x_binsize


plt.figure(1)
plt.xlabel("car's x-position")
plt.ylabel("Fitted Reward")

if original:
	plt.title("Reward graph of original LP")
else:
	plt.title("Reward graph of modified LP")
plt.plot(X,R)
plt.show()

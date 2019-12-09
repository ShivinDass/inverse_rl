import random
from scipy.stats import norm
from scipy.optimize import linprog
import lp_helper
from math import exp,sqrt,pi
import seaborn as sns
import matplotlib.pyplot as plt


def calcPDF(a,m,s):
	a=(a-m)/s
	return (exp(-(a)**2/(2))/(sqrt(2*pi)))/s



#A=(0 UP), (1 DOWN), (2 LEFT), (3 RIGHT)
def move(i,j,a):
	if a=="L":
		i = i + random.uniform(-0.1,0.1)
		j = j - 0.2 + random.uniform(-0.1,0.1)

	elif a=="R":
		i = i + random.uniform(-0.1,0.1)
		j = j + 0.2 + random.uniform(-0.1,0.1)

	elif a=="U":
		i = i - 0.2 + random.uniform(-0.1,0.1)
		j = j + random.uniform(-0.1,0.1)

	else:
		i = i + 0.2 + random.uniform(-0.1,0.1)
		j = j + random.uniform(-0.1,0.1)

	aborbingState=False
	if i>=0.8 and j>=0.8:
		aborbingState=True

	if i<0.0:
		if j<0.0:
			i,j = 0.0,0.0
		elif j>1.0:
			i,j = 0.0,1.0
		else:
			i,j = 0.0,j
	elif i>1.0:
		if j<0.0:
			i,j = 1.0,0.0
		elif j>1.0:
			i,j = 1.0,1.0
		else:
			i,j = 1.0,j
	else:
		if j<0.0:
			i,j = i,0.0
		elif j>1.0:
			i,j = i,1.0
		else:
			i,j = i,j

	return i,j,aborbingState


def getMonteCarloTrajectories(p,length,count):
	traj=[]

	for c in range(count):
		i=0
		j=0
		tr=[]
		tr.append([i,j])
		for l in range(length-1):
			action=p[min(49,round(i*50))][min(49,round(i*50))]
			
			i,j,absS=move(i,j,action)
			
			tr.append([i,j])
			if absS:
				break
		traj.append(tr)
	return traj


def valFuncGen(traj):
	val_func=[]
	scale=0.5
	for i in x_basis_mean:
		for j in y_basis_mean:
			#print(i,j)
			v=0
			for tr in range(len(traj)):
				g=1
				for t in traj[tr]:
					y=t[0]
					x=t[1]

					#x=(x-i)/scale
					#y=(y-j)/scale
					tmpX=calcPDF(x,i,scale) #(exp(-(x)**2/(2))/(sqrt(2*pi)))/scale
					tmpY=calcPDF(y,j,scale) #(exp(-(y)**2/(2))/(sqrt(2*pi)))/scale
					v+=g*tmpX*tmpY#norm.pdf(x,loc=i,scale=0.5)*norm.pdf(y,loc=j,scale=0.5)
					g*=gamma
			v=v/len(traj)
			val_func.append(v)

	return val_func


def solve_lp(pred,other):
	c=[0]*(numOfBasis*numOfBasis) + [-1]*(len(other))
	A=[[0 for i in range(len(c))] for j in range(2*len(other))]
	b=[0]*(2*len(other))
	bound=[(-1,1) for i in range(numOfBasis*numOfBasis)]+[(None,None) for j in range(len(other))]

	for i in range(len(other)):
		A[2*i][numOfBasis*numOfBasis+i]=1
		A[2*i+1][numOfBasis*numOfBasis+i]=1
		for j in range(numOfBasis*numOfBasis):
			# A[2*i][numOfBasis*numOfBasis+i]=1
			# A[2*i+1][numOfBasis*numOfBasis+i]=1

			A[2*i][j]=-1*pred[j]+other[i][j]
			A[2*i+1][j]=4*(-1*pred[j]+other[i][j])

	res = linprog(c,A_ub=A,b_ub=b,bounds=bound)
	# c=[0]*(numOfBasis*numOfBasis)
	# for i in range(numOfBasis*numOfBasis):
	# 	c[i] = -1*len(other)*pred[i]
	# 	for o in other:
	# 		c[i]=c[i]+o[i]

	# b=[(-1,1) for i in range(numOfBasis*numOfBasis)]

	# res=linprog(c,bounds=b)
	#print(res['x'])
	return res['x']


def makeNewReward(V,alph):
	R=[[0 for ii in range(N)] for jj in range(N)]
	scale=0.5

	x=1/(N*2)
	i=0
	while i<N:
		y=1/(N*2)
		j=0
		while j<N:
			for xm in range(len(x_basis_mean)):
				for ym in range(len(y_basis_mean)):	
					tmpX=calcPDF(x,x_basis_mean[xm],scale)
					tmpY=calcPDF(y,y_basis_mean[ym],scale)
					R[j][i]=alph[xm*len(x_basis_mean)+ym]*tmpX*tmpY

			y+=1/N
			j+=1
		
		x+=1/N
		i+=1
		

	return R



def showPolicy(P):
	disp=["UP","DOWN","LEFT","RIGHT"]
	print("POLICY:")
	for i in range(N):
		for j in range(N):
			print(P[i][j],end=" ")
		print()

def showReward(R):
	print("REWARD")
	for i in range(N):
		for j in range(N):
			print("%.1f" % R[i][j],end=" ")
		print()

def comparePolicy(P,P1):
	same=0
	for i in range(N):
		for j in range(N):
			if P1[i][j] in P[i][j]:
				same+=1
	return same/(N*N)

random.seed(0)
N=50
numOfBasis=15
gamma=0.9

alphas=[0]*(numOfBasis*numOfBasis)
x_basis_mean=[i/(numOfBasis-1) for i in range(numOfBasis-1)]+[1.0]
y_basis_mean=[i/(numOfBasis-1) for i in range(numOfBasis-1)]+[1.0]
# print(x_basis_mean,y_basis_mean)

policy_predicted, equiv_poliy=lp_helper.getOptimalPolicy(N,None)
value_func_predicted=valFuncGen(getMonteCarloTrajectories(policy_predicted,30,1000))
showPolicy(policy_predicted)
#exit(0)

random_policy=[[random.choice(["U","D","L","R"]) for i in range(N)] for j in range(N)]
policy_set=[random_policy]
print(comparePolicy(policy_predicted,random_policy))

trajectory_set=[]
value_function_set=[]

for T in range(1):
	print("Generating Trajectories")
	trajectory_set.append(getMonteCarloTrajectories(policy_set[-1],30,1000))
	print("Computing Value Functions")
	value_function_set.append(valFuncGen(trajectory_set[-1]))


	print("Solving LP")
	alphas=solve_lp(value_func_predicted,value_function_set)
	#print(alphas)

	R=makeNewReward(value_func_predicted,alphas)
	showReward(R)

	P,e=lp_helper.getOptimalPolicy(N,R)
	showPolicy(P)
	print(comparePolicy(policy_predicted,P))
	policy_set.append(P)
	sns.heatmap(R, xticklabels=False, yticklabels=False)
	plt.title("Continuous Gridworld Retrieved Reward")
	plt.xlabel("States in x-dimension")
	plt.ylabel("States in y-dimension")
	plt.show()



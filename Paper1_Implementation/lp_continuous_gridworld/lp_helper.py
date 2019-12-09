import numpy as np
import copy
import random


def getExpectedValue(V,i,j,a):
	global N
	global rewardRegion
	global error
	global moveLength
	global gamma
	global R

	averagingOver=error*2+1
	avg=0.0
	if a=="L":
		for k in range(j-moveLength-error,j-moveLength+error+1):
			move=k
			if move<0:
				move=0

			r=R[i][j]
			avg+=(r+gamma*V[i][move])

	elif a=="R":
		for k in range(j+moveLength-error,j+moveLength+error+1):
			move=k
			if move>=N:
				move=N-1

			r=R[i][j]
			avg+=(r+gamma*V[i][move])

	elif a=="U":
		for k in range(i-moveLength-error,i-moveLength+error+1):
			move=k
			if move<0:
				move=0

			r=R[i][j]
			avg+=(r+gamma*V[move][j])

	else:
		for k in range(i+moveLength-error,i+moveLength+error+1):
			move=k
			if move>=N:
				move=N-1

			r=R[i][j]
			avg+=(r+gamma*V[move][j])

	return avg/averagingOver



def getStateAndReward(i,j,a):
	
	if a=="U":
		#print("U")
		i = i + random.randint(-1*error,error)
		j = j - moveLength + random.randint(-1*error,error)

	elif a=="D":
		#print("D")
		i = i + random.randint(-1*error,error)
		j = j + moveLength + random.randint(-1*error,error)

	elif a=="L":
		#print("L")
		i = i - moveLength + random.randint(-1*error,error)
		j = j + random.randint(-1*error,error)

	else:
		#print("R")
		i = i + moveLength + random.randint(-1*error,error)
		j = j + random.randint(-1*error,error)

	r=0.0
	if i>=rewardRegion and j>=rewardRegion:
		r=1.0
	
	#if r==1:
	#print(i,j,r)

	if i<0:
		if j<0:
			return 0,0,r
		elif j>=N:
			return 0,N-1,r
		else:
			return 0,j,r
	elif i>=N:
		if j<0:
			return N-1,0,r
		elif j>=N:
			return N-1,N-1,r
		else:
			return N-1,j,r
	else:
		if j<0:
			return i,0,r
		elif j>=N:
			return i,N-1,r
		else:
			return i,j,r


def updateValues(V,policy):
	global N
	global rewardRegion
	global error
	global moveLength
	global gamma
	global R

	v=[[0.0 for i in range(N)] for j in range(N)]
	for i in range(N):
		for j in range(N):
			v[i][j]=getExpectedValue(V,i,j,policy[i][j]) 
	return v


def updatePolicy(V):
	global N
	global rewardRegion
	global error
	global moveLength
	global gamma
	global R

	p=[["U" for i in range(N)] for j in range(N)]
	for i in range(N):
		for j in range(N):
			k="U"
			maxim=getExpectedValue(V,i,j,"U")

			for x in ["D","L","R"]:
				tmpExp=getExpectedValue(V,i,j,x)
				if tmpExp>maxim:
					maxim=tmpExp
					k=x
			p[i][j]=k
	return p


def getEquivalentPolicy(V):
	global N
	global rewardRegion
	global error
	global moveLength
	global gamma
	global R

	p=[["U" for i in range(N)] for j in range(N)]
	for i in range(N):
		for j in range(N):
			k={"U"}
			maxim=getExpectedValue(V,i,j,"U")

			for x in ["D","L","R"]:
				tmpExp=getExpectedValue(V,i,j,x)
				if abs(tmpExp-maxim)<0.0001:
					maxim=max(tmpExp,maxim)
					k.add(x)
				elif tmpExp>maxim:
					maxim=tmpExp
					k={x}
			p[i][j]=k
	return p


def showValue(Val):
	global N
	global rewardRegion
	global error
	global moveLength
	global gamma
	global R

	print("STATE-VALUE FUNCTION")	
	for i in range(N):
		for j in range(N):
			print("%.1f" % Val[i][j],end=" ")
		print()


def showPolicy(P):
	global N
	global rewardRegion
	global error
	global moveLength
	global gamma
	global R

	disp=["UP","DOWN","LEFT","RIGHT"]
	print("POLICY:")
	for i in range(N):
		for j in range(N):
			out=""
			for k in P[i][j]:
				out+=k
			print(out,end=" ")
		print()






def getOptimalPolicy(n,r):
	global N
	global rewardRegion
	global error
	global moveLength
	global gamma
	global R

	N=n
	if r!=None:
		R=r
	else:
		for i in range(N):
			for j in range(N):
				if i>=rewardRegion and j>=rewardRegion:
					R[i][j]=1.0

	V=[[0.0 for i in range(N)] for j in range(N)] 
	policy=[["D" for i in range(N)] for j in range(N)] #U:0, D:1, L:2, R:3


	t=0
	while 1>0:
		t+=1

		tt=0
		#Policy Evaluation
		while 1>0:
			tt+=1
			_V=updateValues(copy.deepcopy(V),policy)
		
			flag_break=True
			for i in range(N):
				for j in range(N):
					if abs(V[i][j]-_V[i][j])>0.001:
						flag_break=False
						break
				if flag_break==False:
					break

			if flag_break:
				break
			else:
				V=copy.deepcopy(_V)

		#Policy Improvement
		_policy=updatePolicy(V)
		

		if policy==_policy:
			break
		else:
			policy=copy.deepcopy(_policy)


	equiv_policy=getEquivalentPolicy(V)
	#showValue(V)
	return policy,equiv_policy

N=50
rewardRegion=round(0.8*N)
error=round(0.1*N)
moveLength=round(0.2*N)
gamma=0.9
R=[[0.0 for i in range(N)] for j in range(N)]

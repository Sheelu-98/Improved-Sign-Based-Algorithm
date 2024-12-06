import numpy as np
import math as m
import matplotlib.pyplot as plt
#from neupy.optimizations import wolfe
def transfer(x):
    return 1/(1+np.exp(-x))

def deriv(o):
    f = np.matrix(np.zeros((len(o),len(o))))
    for i in range(len(o)):
        f[i,i] = o[i,0]*(1-o[i,0])
    return f

def sgn(k):
    if k<0:
        return -1
    elif k>0:
        return 1
    elif k==0:
        return 0

i = np.matrix([[1.,1.],[1.,0.],[0.,1.],[0.,0.]])
o = np.matrix([[0.],[1.],[1.],[0.]])
wt1 = np.matrix(np.ones((2,2)))
wt2 = np.matrix(np.ones((1,2)))
b1 = np.matrix(np.ones((2,1)))
b2 = np.matrix(np.ones((1,1)))
y1 = [[],[],[],[]]
y2 = [[],[],[],[]]
e = [[],[],[],[]]

dw2 = [[],[],[],[]]
db2 = [[],[],[],[]]
db1 = [[],[],[],[]]
dw1 = [[],[],[],[]]
lr = [[],[],[],[]]
q = [[],[],[],[]]
E = []
lr = [np.ones((1,1))*0.1,np.ones((1,2))*0.1,np.ones((2,1))*0.1,np.empty((2,2))*0.1]
q_prev = [np.ones((1,1)),np.ones((1,2)),np.ones((2,1)),np.ones((2,2))]
dell = [np.ones((1,1)),np.ones((1,2)),np.ones((2,1)),np.ones((2,2))]
dell_prev = [np.ones((1,1)),np.ones((1,2)),np.ones((2,1)),np.ones((2,2))]

for z in range(100):
    for j in range(len(i)):
        y1[j] = transfer(wt1*np.transpose(i[j]) + b1)
        y2[j] = transfer(wt2*y1[j] + b2)
        e[j] =  o[j] - y2[j]
        db2[j] = -2*deriv(y2[j])*e[j]
        dw2[j] = db2[j]*np.transpose(y1[j])
        db1[j] = deriv(y1[j])*np.transpose(wt2)*db2[j]
        dw1[j] = db1[j]*i[j]
    
    for d in range(4):
        p = [db2,dw2,db1,dw1]
        q[d] = np.sum(p[d],axis=0)/4
    
    param = [b2,wt2,b1,wt1]
    E.append(np.sum([e[d1]**2 for d1 in range(4)]))
    if E[z] <= E[z-1]:
        for d2 in range(4):
            for x in range(np.shape(q[d2])[0]):
                for y in range(np.shape(q[d2])[1]):
                    if q_prev[d2][x,y]*q[d2][x,y] > 0:
                        lr[d2][x,y] = min(lr[d2][x,y]*1.2,50)
                        dell[d2][x,y] = -1* sgn(q[d2][x,y])*lr[d2][x,y] 
                        param[d2][x,y] += dell[d2][x,y] 
                        q_prev[d2][x,y] = q[d2][x,y]
                        dell_prev[d2][x,y] = dell[d2][x,y]
                    elif q_prev[d2][x,y]*q[d2][x,y] <0:
                        lr[d2][x,y] = max(lr[d2][x,y]*0.5,0.000001)
                        q_prev[d2][x,y] = 0
                    elif q_prev[d2][x,y]*q[d2][x,y] == 0:
                        dell[d2][x,y] = -1* sgn(q[d2][x,y])*lr[d2][x,y]
                        param[d2][x,y] += dell[d2][x,y] 
                        q_prev[d2][x,y] = q[d2][x,y]
                        dell_prev[d2][x,y] = dell[d2][x,y]
        c = 1
    elif E[z] > E[z-1]:
        for d2 in range(4):
            for x in range(np.shape(q[d2])[0]):
                for y in range(np.shape(q[d2])[1]):
                    param[d2][x,y] += m.pow(0.5,c)*dell_prev[d2][x,y]
                    c += 1

h = np.matrix([0,0])
n1 = transfer(wt1*np.transpose(h) + b1)
n2 = transfer(wt2*n1 + b2)
print (n2)

plt.plot(E)
plt.ylabel('Sum of Squared Error')
plt.xlabel('Epoch')
plt.title('Performance Plot of JRprop')

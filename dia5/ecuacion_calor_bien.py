import numpy as np
import math
import matplotlib.pyplot as plt



b= float(input ("b: ")) #5
d= float(input ("d: ")) #10
N= int(input ("N: ")) #40
M= int(input ("M: "))#400
h= b/N
K= d/M
w= np.zeros((M+1, N+1))
v=float(input("conductividad "))


def f(x):
    return math.exp(-(h*i-b/2)**2)

for j in range (1, M): 
    w[j][0]= 0
    w[j][N]= 0
    
for i in range (1, N):
    w[0][i]= f(h*i)
    
    
for j in range(M):
    for i in range(1, N):
        w[j+1][i]= (1-2*K*v**2/h**2)*w[j][i]+(K*v**2/h**2)*(w[j][i+1]+w[j][i-1])
        
        
X= np.linspace(0, b, N+1)
Y= np.linspace(0, d, M+1)
X, Y= np.meshgrid(X, Y)

fig= plt.figure()
ax= fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, w, cmap='viridis')

ax.set_xlabel('Eje X')
ax.set_ylabel('Eje Y')

plt.show()
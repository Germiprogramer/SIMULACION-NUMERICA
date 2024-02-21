import numpy as np

a = float(input("a: ")) #0
b = float(input("b: ")) #1
c = float(input("c: ")) #0
d = float(input("d: ")) #1
N = int(input("N: ")) #50
M = int(input("M: ")) #50
h = (b-a)/N
k = (d-c)/M
w = np.zeros((N+1, M+1))

def f(i,j):
    return 0

for i in range(1,N):
    w[i][0] = 0
    w[i][M] = (a+i*h)**2
for j in range(1,M):
    w[0][j] = 0
    w[N][j] = (b+j*k)**2

for k in range(100): 
    for i in range(1,N):
        for j in range(1,M):
            w[i][j] = (k**2*2*(w[i+1][j]+w[i-1][j])+h**2*(w[i][j+1]+w[i][j-1])-(h*k)**2*f[i][j])/(2*(h**2+k**2))


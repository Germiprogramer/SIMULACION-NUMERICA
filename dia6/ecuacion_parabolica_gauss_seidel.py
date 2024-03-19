#codigo que no esta bien

import numpy as np
import math
import matplotlib.pyplot as plt

b = float(input("b: ")) # example: 5
d = float(input("d: ")) # example: 5
N = int(input("N: ")) # example: 40
M = int(input("M: ")) # example: 40
h = b / N
K = d / M
w = np.zeros((M + 1, N + 1))
a = float(input("conductivity a: ")) # the thermal diffusivity, must be defined

lambda_ = a**2 * K / h**2

def f(x, b, i, h):
    return math.exp(-(h * i - b / 2) ** 2)

for i in range(1, N):
    w[0][i] = f(h * i, b, i, h)

# Gauss-Seidel iteration
for p in range(100):
    for j in range(1, M): # ensure j starts at 1 to avoid index error for j-1
        for i in range(1, N):
            w[j][i] = (lambda_ * (w[j][i + 1] + w[j - 1][i] + w[j][i - 1])) / (1 + 2 * lambda_)

# Set boundary conditions
for j in range(M + 1):
    w[j][0] = 0
    w[j][N] = 0

X, Y = np.meshgrid(np.linspace(0, b, N + 1), np.linspace(0, d, M + 1))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, w, cmap='viridis')
ax.set_xlabel('Eje X')
ax.set_ylabel('Eje Y')
plt.show()

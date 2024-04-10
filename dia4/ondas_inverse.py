import numpy as np
import matplotlib.pyplot as plt

# Entrada de parámetros
a = 0
b = float(input('Ingresar valor de b: '))
c = 0
d = float(input('Ingresar valor de d: '))
N = int(input('Ingresar valor de N: '))
M = int(input('Ingresar valor de M: '))

# Cálculo de pasos
h = (b - a) / N
k = (d - c) / M


vmax = h/k  # (BM)/(Nd)


print("Velocidad máxima: ", vmax)

v = float(input('Ingresar valor de velocidad: '))

p = v*k/h

# Inicialización de la matriz w con dimensiones correctas
w = np.zeros((N+1, M+1))  # +1 para incluir los bordes


def f(x):
    return x*(b-x)

def g(x):
    return 0


for i in range(1, N):
    w[i][0] = f(h*i)  # Frontera inferior
    # Frontera superior, recordar que xi= xo(a) + ih
    w[i][1] = w[0][i] + k*g(h*i)

for j in range(1, M):
    w[0][j] = 0  # Frontera izquierda, recordar que yi= yo(c) + kj
    w[N][j] = 0  # Frontera derecha
    
# Iteraciones para la solución
for j in range(1, M):
    for i in range(1, N):
        w[i][j+1] = 2*(1-p**2)*w[i][j] + (p**2) * \
            (w[i+1][j] + w[i-1][j]) - w[i][j-1]


# Set up the figure and 3D axes
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Create grid
x = np.linspace(a, b, N+1)
y = np.linspace(c, d, M+1)
X, Y = np.meshgrid(x, y)

# Plot the surface
ax.plot_surface(X, Y, w.T, cmap='viridis')

# Set labels and show plot
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('U axis')
ax.set_title('Gauss-Seidel Solution for Laplace Equation with Given Boundary Conditions')

plt.show()
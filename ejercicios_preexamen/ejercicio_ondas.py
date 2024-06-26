#Para ecuaciones hiperbólicas

'''
dividir la malla en subdivisiones de tamaño h y k
aproximar las derivadas parciales de segundo orden por el polinomio de Taylor
sustituir las derivadas parciales por las aproximaciones
resolver la ecuación resultante para el nodo central
repetir el proceso para cada nodo de la malla
'''

import numpy as np
import matplotlib.pyplot as plt

# Entrada de parámetros
a = 0
b = float(input('Ingresar valor de b: ')) #5
c = 0
d = float(input('Ingresar valor de d: ')) #10
N = int(input('Ingresar valor de N: ')) #40
M = int(input('Ingresar valor de M: ')) #400

# Cálculo de pasos
h = (b - a) / N
k = (d - c) / M


vmax = h/k  # (BM)/(Nd)


print("Velocidad máxima: ", vmax)

v = float(input('Ingresar valor de velocidad: ')) #0.5

p = v*k/h

# Inicialización de la matriz w con dimensiones correctas
w = np.zeros((M+1, N+1))  # +1 para incluir los bordes


def f(x):
    return 0

def g(x):
    return 0


for j in range(1, M):
    w[j][0] = 3 * np.cos(k*j)  # Frontera izquierda, recordar que yi= yo(c) + kj
    w[j][N] = 0  # Frontera derecha


for i in range(1, N):
    w[0][i] = f(h*i)  # Frontera inferior
    # Frontera superior, recordar que xi= xo(a) + ih
    w[1][i] = w[0][i] + k*g(h*i)


# Iteraciones para la solución
for j in range(1, M):
    for i in range(1, N):
        w[j + 1][i] = 2*(1-p**2)*w[j][i] + (p**2) * \
            (w[j][i + 1] + w[j][i-1]) - w[j-1][i]


# Crear una malla de coordenadas para graficar
x = np.linspace(a, b, N+1)
y = np.linspace(c, d, M+1)
X, Y = np.meshgrid(x, y)

# Crear la figura y el eje para la gráfica 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Graficar la superficie
# Transponer w para que coincida con las dimensiones de X y Y
surf = ax.plot_surface(X, Y, w, cmap='viridis', edgecolor='none')
ax.set_title('Solución de la EDP de Ondas')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('w')

# Añadir barra de colores para la escala
fig.colorbar(surf)
plt.show()

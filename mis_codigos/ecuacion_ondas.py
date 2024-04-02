import numpy as np
import matplotlib.pyplot as plt

# Entrada de parámetros
a = 0
c = 0
b = float(input('Ingresar valor de b: '))
d = float(input('Ingresar valor de d: '))
N = int(input('Ingresar valor de N: '))
M = int(input('Ingresar valor de M: '))
v = float(input('Ingresar valor de velocidad: '))

# Cálculo de pasos
h = b / N
k = d / M

# Inicialización de la matriz w con dimensiones correctas
w = np.zeros((N+1, M+1))  # +1 para incluir los bordes

# Función f(i, j) como fuente


def f(x):  # Laplaciano de u
    return x*(b-x)

def g(x):  # Condición inicial
    return 0


for i in range(N):
    w[i][0] = f(0)  # Frontera inferior
    w[i][1] = w[i][0] + k*g(i)  # Frontera inferior en t=1

for j in range(M):
    w[0][j] = 0
    w[N][j] = 0


# Iteraciones para la solución

for i in range(1, N):
        for j in range(1, M):
            w[i][j+1] = 2*(1-(v**2 * k**2)/(h**2))*w[i][j] + ((v*k)/h)**2 * (w[i+1][j] + w[i-1][j]) - w[i][j-1]

# Crear una malla de coordenadas para graficar
x = np.linspace(a, b, N+1)
y = np.linspace(c, d, M+1)
X, Y = np.meshgrid(x, y)

# Crear la figura y el eje para la gráfica 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Graficar la superficie
# Transponer w para que coincida con las dimensiones de X y Y
surf = ax.plot_surface(X, Y, w.T, cmap='viridis', edgecolor='none')
ax.set_title('Solución de la EDP con diferencias finitas')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('w')

# Añadir barra de colores para la escala
fig.colorbar(surf)
plt.show()
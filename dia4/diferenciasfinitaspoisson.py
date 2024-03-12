import numpy as np
import matplotlib.pyplot as plt

# Entrada de parámetros
a = float(input('Ingresar valor de a: '))
b = float(input('Ingresar valor de b: '))
c = float(input('Ingresar valor de c: '))
d = float(input('Ingresar valor de d: '))
N = int(input('Ingresar valor de N: '))
M = int(input('Ingresar valor de M: '))

# Cálculo de pasos
h = (b - a) / N
k = (d - c) / M

# Inicialización de la matriz w con dimensiones correctas
w = np.zeros((N+1, M+1))  # +1 para incluir los bordes

# Función f(i, j) como fuente


def f(i, j):  # Laplaciano de u
    return 0


for i in range(N):
    w[i][0] = 0  # Frontera inferior
    w[i][M] = 0  # Frontera superior, recordar que xi= xo(a) + ih

for j in range(M):
    w[0][j] = c + k*j  # Frontera izquierda, recordar que yi= yo(c) + kj
    w[N][j] = c + k*j  # Frontera derecha


# Iteraciones para la solución
for p in range(100):
    for i in range(1, N):
        for j in range(1, M):
            w[i][j] = (k**2*(w[i+1][j] + w[i-1][j]) + h**2*(w[i]
                       [j+1] + w[i][j-1]) - h**2*k**2*f(i, j))/(2*(h**2 + k**2))


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

# ? Ejercicio 1 EDP Poisson condiciones frontera = 0,0,0,1 siendo a = 0 b = 1 c = 0 d = 1 N = 100 M = 100

# ? Ejercicio 2 EDP condiciones frontera = 0,(a+h*i)**2 ,1-(c+k*j)**2,1 siendo a = 0 b = 1 c = 0 d = 1 N = 40 M = 40

# ? Ejercicio 3 EDP condiciones frontera = 0,0,0,0 siendo a = 0 b = 1 c = 0 d = 1 N = 100 M = 100 y el Laplaciano de u = xy(1-x)(1-y)

# ? Ejercicio 4 EDP condiciones frontera = 0,0,y,y siendo a = 0 b = pi c = 0 d = pi N = 100 M = 100 y el Laplaciano de u = 0
## S E I D E L


import numpy as np  # Descomenta esta línea para importar numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Conversión de entradas a números de punto flotante
a = float(input("Ingrese el valor de a: ")) #0
b = float(input("Ingrese el valor de b: ")) #5
c = float(input("Ingrese el valor de c: ")) #0
d = float(input("Ingrese el valor de d: ")) #5
N = int(input("Ingrese el valor de N: ")) #40
M = int(input("Ingrese el valor de M: ")) #400
alpha = float(input("Ingrese el valor de alpha: ")) #0.4

# Calculando los tamaños de paso
h = (b - a) / N
k = (d - c) / M
# Definición de parámetros
alpha_sq = (alpha) ** 2

alpha_max = h/np.sqrt(2*k)
print(f'Alpha máxima: {alpha_max}')
# Inicialización de la matriz w
w = [[0] * (M + 1) for _ in range(N + 1)]

# Aplicando las condiciones de contorno
for j in range(M + 1):
    w[0][j] = 0  # u(0,t)= 0
    w[N][j] = 0  # u(L,t)= 0

# Aplicando las condiciones iniciales
def f(xi):
    return np.exp(-(xi-(b/2))**2) # Define tu función f(xi) aquí

def g(xi):
    return 0  # Define tu función g(xi) aquí

for i in range(N + 1):
    w[i][0] = f(a + i * h)  # u(x,0)=f(x)
    w[i][1] = w[i][0] + k * g(a + i * h)  # du(x,0)/dt=g(x)

# Calculando los valores de w utilizando la ecuación proporcionada
for p in range(100):
    for j in range(1, M):
        for i in range(1, N):
            w[i][j] = ((k/h**2)*(w[i+1][j] + w[i-1][j])+(1+h*i)*w[i][j-1])/(1+h*i+2*(k/h**2))
# Graficando la solución
# Crear una malla tridimensional
X = np.linspace(a, b, N+1)
Y = np.linspace(c, d, M+1)
X, Y = np.meshgrid(X, Y)

# Crear la figura
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Graficar la superficie
surf = ax.plot_surface(X, Y, np.array(w).T, cmap='viridis', edgecolor='none')
ax.set_xlabel('X')
ax.set_ylabel('T')
ax.set_zlabel('w')
ax.set_title('Solución de la ecuación en 3D')

# Agregar una barra de colores
fig.colorbar(surf, shrink=0.5, aspect=5)

# Mostrar el gráfico
plt.show()

# u = fuente de calor
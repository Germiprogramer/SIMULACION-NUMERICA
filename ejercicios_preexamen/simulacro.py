import numpy as np
import matplotlib.pyplot as plt

# Entrada de parámetros
a = 0
c = -1
b = float(input('Ingresar valor de b: ')) # 1
d = float(input('Ingresar valor de d: ')) # 0
N = int(input('Ingresar valor de N: ')) # 30
M = int(input('Ingresar valor de M: ')) # 30

#Laplace, ondas y parabolico

# Cálculo de pasos
#x = a + h*i
h = (b-a) / N
#y = c  + k*j
k = (d-c) / M

# Inicialización de la matriz w con dimensiones correctas
w = np.zeros((N+1, M+1))  # +1 para incluir los bordes

for i in range(N):
    w[i][0] = -5
    w[i][M] = 10*(h*i)*(1-h*i) # Frontera inferior
    
for j in range(M):
    w[0][j] = 5*(k*j+c)
    #escribeme sen(2*pi*y)
    w[N][j] = 5*(np.sin(2*np.pi*(k*j+c)))


# Iteraciones para la solución
for p in range(100):
    for i in range(1, N):
        for j in range(1, M):
            w[i][j] = (h*i*(h**2)*(w[i][j+1]+w[i][j-1]) - (c+j*k)*(k**2)*(w[i+1][j]+w[i-1][j]) ) / (2*(h*i*(h**2) - (c+j*k)*(k**2)))
        
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
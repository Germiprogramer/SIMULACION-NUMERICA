import numpy as np
import matplotlib.pyplot as plt

# Configuración de parámetros iniciales basados en la EDP y las condiciones de frontera dadas
b = 5.0  # Límite superior de x
d = 5.0  # Límite superior de t
N = 40   # Número de puntos en el espacio
M = 40   # Número de puntos en el tiempo
h = b / N  # Tamaño del paso en x
k = d / M  # Tamaño del paso en t
a = 0.5  # Valor de conductividad

# Matriz de solución con condiciones iniciales y de frontera aplicadas
w_rd = np.zeros((M + 1, N + 1))
for i in range(1, N):
    w_rd[0, i] = np.exp(-(h*i - 2.5)**2)
w_rd[:, 0] = w_rd[:, N] = 0

# Parámetro lambda para el método de diferencias finitas
lambda_ = (a**2) * k / (h**2)

# Implementación del método de diferencias finitas implícitas
for j in range(1, M+1):
    # Creamos las matrices tridiagonales para el sistema lineal de la iteración j
    A = np.zeros((N-1, N-1))
    b = np.zeros(N-1)
    for i in range(1, N):
        A[i-1, i-1] = 1 + 2*lambda_ + k*(1+h*i)  # Diagonal principal
        if i > 1:
            A[i-1, i-2] = -lambda_  # Diagonal inferior
        if i < N-1:
            A[i-1, i] = -lambda_  # Diagonal superior
        # Término independiente con el término de reacción incluido
        b[i-1] = k * w_rd[j-1, i] + k*(1+h*i)*w_rd[j-1, i] + w_rd[j-1, i]

    # Resolver el sistema lineal
    w_rd[j, 1:N] = np.linalg.solve(A, b)

# Malla para gráfica
X_rd, Y_rd = np.meshgrid(np.linspace(0, b, N + 1), np.linspace(0, d, M + 1))

# Crear gráfica
fig_rd = plt.figure()
ax_rd = fig_rd.add_subplot(111, projection='3d')
ax_rd.plot_surface(X_rd, Y_rd, w_rd, cmap='viridis')
ax_rd.set_xlabel('X axis')
ax_rd.set_ylabel('Y axis')
ax_rd.set_zlabel('U')
ax_rd.set_title('Reacción-Difusión EDP Solución')
plt.show()

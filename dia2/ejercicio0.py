#Poisson creo

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

a, b = 0, 1  # x boundaries
c, d = 0, 1  # y boundaries
N, M = 20, 20  # Number of grid points
h = (b-a)/N  # Grid spacing in x
k = (d-c)/M  # Grid spacing in y

w = np.zeros((N+1, M+1))

# Apply the new boundary conditions
for i in range(N+1):
    w[i, 0] = 0  # u(x, 0) = 0
    w[i, M] = 0  # u(x, 1) = 0

for j in range(M+1):
    w[0, j] = w[N, j]  # u(0, y) = u(1, y)
    w[N, j] = w[0, j]  # u(1, y) = u(0, y)

# Gauss-Seidel iteration
# Assuming convergence within 100 iterations for demonstration purposes
for _ in range(100):
    for i in range(1, N):
        for j in range(1, M):
            # Apply Gauss-Seidel update
            w[i, j] = (h**2 * (w[i, j+1] + w[i, j-1]) + k**2 * (w[i+1, j] + w[i-1, j])) / (2*(h**2 + k**2))

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
ax.set_title('Gauss-Seidel Solution for Laplace Equation with New Boundary Conditions')

plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

a, b = 0, 1  # x boundaries
c, d = 0, 1  # y boundaries
N, M = 20, 20  # Number of grid points
h = (b-a)/N  # Grid spacing in x
k = (d-c)/M  # Grid spacing in y

w = np.zeros((N+1, M+1))

# Gauss-Seidel iteration with the source term
# Assuming convergence within 10000 iterations for demonstration purposes
for _ in range(10000):
    for i in range(1, N):
        for j in range(1, M):
            # Calculate the source term
            f_ij = (a + i*h) * (c + j*k) * (1 - (a + i*h)) * (1 - (c + j*k))
            # Apply Gauss-Seidel update with source term
            w[i, j] = ((h**2 * (w[i, j+1] + w[i, j-1]) + k**2 * (w[i+1, j] + w[i-1, j])) / (2*(h**2 + k**2))) - (h**2 * k**2 * f_ij / (2*(h**2 + k**2)))

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
ax.set_title('Gauss-Seidel Solution for Poisson Equation with Source Term')

plt.show()

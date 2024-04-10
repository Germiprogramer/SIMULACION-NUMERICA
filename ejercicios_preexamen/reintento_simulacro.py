#ecuaciones elipticas

'''
esto se hace por Gauss-Seidel
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the boundaries of the domain
a = 0
b = 1 # x boundaries
c = -1
d = 0  # y boundaries
N = 30
M = 30 # Number of grid points

#x = a + h*i
#y = c + k*j

h = (b-a)/N  # Grid spacing in x
k = (d-c)/M  # Grid spacing in y

w = np.zeros((N+1, M+1))

# Apply the boundary conditions
for i in range(N):
    w[i][0] = 10*(a+h*i)*(1-(a+h*i)) # Frontera inferior
    w[i][M] = -5 # Frontera superior
    
for j in range(M):
    w[0][j] = 5*(c+k*j)
    w[N][j] = 5*(np.sin(2*(np.pi)*(c+k*j)))

# Gauss-Seidel iteration
# Assuming convergence within 10000 iterations for demonstration purposes
for _ in range(100):
    for i in range(1, N):
        for j in range(1, M):
            # Apply Gauss-Seidel update
            w[i][j] = (((c+k*j)/(h**2))*(w[i+1, j] + w[i-1, j]) - ((a+h*i)/(k**2))*(w[i, j+1] + w[i, j-1]))/(2*((c+k*j)/(h**2) - (a+h*i)/(k**2)))

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
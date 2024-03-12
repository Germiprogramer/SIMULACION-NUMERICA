import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

tiempo = 10

v = 0.5 # Wave speed
a, b = 0, 5  # x boundaries
c, d = 0, 10  # y boundaries
N, M = 40, 400  # Number of grid points
h = (b-a)/N  # Grid spacing in x
k = (d-c)/M  # Grid spacing in y
p = (v*k)/h

w = np.zeros((N+1, M+1))

def f(x):
     return x*(b-x)
def g(x):
    return 0

# Apply the new boundary conditions
for i in range(N+1):
    w[i, 0] = 0
    w[i, M] = 0
for j in range(M+1):
    w[0, j] = f(h*i)
    w[N, j] = w[0][i]+k*g(h*i)



# Gauss-Seidel iteration
# Assuming convergence within 100 iterations for demonstration purposes

for i in range(1, N):
    for j in range(1, M):
            # Apply Gauss-Seidel update
        w[i, j+1] = 2*(1-p**2)*w[i, j] + (p**2)*(w[i+1, j] - w[i-1, j])-w[i, j-1]



# Set up the figure and 3D axes
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Create grid
x = np.linspace(0, b, N+1)
X, Y = np.meshgrid(x, y)

# Plot the surface
ax.plot_surface(X, Y, w.T, cmap='viridis')

# Set labels and show plot
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('U axis')
ax.set_title('Gauss-Seidel Solution for Laplace Equation with New Boundary Conditions')

plt.show()

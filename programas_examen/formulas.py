derx = w[i+1][j] - 2*w[i][j] + w[i-1][j] / h**2
dery = w[i][j+1] - 2*w[i][j] + w[i][j-1] / k**2
dercruz = (w[i+1][j+1] + w[i-1][j-1] - w[i-1][j+1] - w[i+1][j-1]) / (4*h*k)

B**2 - 4*A*C 
> Hiperbolica
= Parabolica
< Eliptica
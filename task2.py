import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the weights
w = np.array([0, 2.5, -0.5])

# Generate the input space grid
x1 = np.arange(-1, 1.05, 0.05)
x2 = np.arange(-1, 1.05, 0.05)
X1, X2 = np.meshgrid(x1, x2)

# Calculate phi(x)
Phi = np.vstack([np.ones_like(X1).ravel(), X1.ravel()**2, X2.ravel()**3]).T

# Function to calculate outputs for a specific sigma value
def generate_data(sigma_squared):
    epsilon = np.random.normal(0, sigma_squared, size=Phi.shape[0])
    t = np.dot( Phi, w) + epsilon
    return t, X1.ravel(), X2.ravel()
# Choose a sigma value for 3D scatter plot
sigma_squared = 0.4
t, flat_X1, flat_X2 = generate_data(sigma_squared)

test_mask = (np.abs(flat_X1) > 0.3) & (np.abs(flat_X2) > 0.3)
train_mask = ~test_mask




# Plotting in 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(flat_X1, flat_X2, t, color='blue', marker='o')
ax.set_title(fr'$\sigma^2$ = {sigma_squared}')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('Output $t_i$')
plt.show()

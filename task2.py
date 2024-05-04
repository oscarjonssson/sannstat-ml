import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import inv

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

def evaluate_models(Phi_train, t_train, Phi_test, t_test):
    # Beräkna viktade skattningar och MSE för Frequentist-modellen
    w_hat = inv(Phi_train.T @ Phi_train) @ Phi_train.T @ t_train
    t_pred_freq = Phi_test @ w_hat
    mse_freq = np.mean((t_test - t_pred_freq) ** 2)
    return mse_freq, w_hat


def plot_data_and_predictions(X1, X2, t, t_pred_freq):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')  # Using one subplot for both

    # Plot data points
    scatter = ax.scatter(X1, X2, t, color='blue', label='Actual Data Points', alpha=0.7)
    
    # Plot predictions
    scatter_pred = ax.scatter(X1, X2, t_pred_freq, color='red', label='pred_freq', alpha=0.7)
    
    ax.set_title("3D Visualization of Actual Data and Predictions")
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('Output $t$')
    ax.legend()
    plt.show()


    
def main():
    # Choose a sigma value for 3D scatter plot
    sigma_squared = 0.4
    t, X1, X2 = generate_data(sigma_squared)

    # Skapa mask för att separera tränings- och testdata baserat på x1 och x2
    test_mask = (np.abs(Phi[:, 1]) > 0.3) & (np.abs(Phi[:, 2]) > 0.3)
    Phi_train, t_train = Phi[~test_mask], t[~test_mask]
    Phi_test, t_test = Phi[test_mask], t[test_mask]

    # Evaluate the model with both training and testing datasets
    mse_freq, w_hat = evaluate_models(Phi_train, t_train, Phi_test, t_test)

    # Beräkna prediktioner för hela datasetet
    t_pred_freq = Phi @ w_hat
    # Visualisera data och prediktioner
    plot_data_and_predictions(X1, X2, t, t_pred_freq)

if __name__ == "__main__":
    main()
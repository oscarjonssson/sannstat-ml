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

def frequentist_regression(Phi_train, t_train, Phi_test, t_test):
    # Beräkna viktade skattningar och MSE för Frequentist-modellen
    w_hat = inv(Phi_train.T @ Phi_train) @ Phi_train.T @ t_train
    t_pred_freq = Phi_test @ w_hat
    mse_freq = np.mean((t_test - t_pred_freq) ** 2)

    return mse_freq, w_hat

def bayesian_regression(Phi_train, t_train, alpha, sigma_squared):
    # Prior mean and covariance matrix of the weights
    prior_mean = np.zeros(Phi_train.shape[1])  # Assuming a mean of 0
    prior_covariance = np.eye(Phi_train.shape[1]) / alpha  # Prior covariance matrix

    # Posterior covariance matrix
    posterior_covariance = inv(inv(prior_covariance) + (1/sigma_squared) * Phi_train.T @ Phi_train)
    # Posterior mean
    posterior_mean = posterior_covariance @ ((1/sigma_squared) * Phi_train.T @ t_train + inv(prior_covariance) @ prior_mean)

    # Predict using the posterior mean
    t_pred_bayes = Phi_train @ posterior_mean
    return t_pred_bayes, posterior_mean, posterior_covariance


def plot_data_and_predictions(X1, X2, t, t_pred, title, alpha, sigma_squared):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot data points
    ax.scatter(X1, X2, t, color='blue', label='Actual Data Points', alpha=0.7)
    
    # Plot predictions
    ax.scatter(X1, X2, t_pred, color='red', label=f'{title} Predictions', alpha=0.7)
    
    if alpha is not None:
        ax.set_title(f"{title} Predictions (α={alpha}, σ²={sigma_squared})")
    else:
        ax.set_title(f"{title} Predictions (σ²={sigma_squared})")
        
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('Output $t$')
    ax.legend()
    plt.show()
   
def main():
    sigma_squared = 0.4  # Variance of the noise
    alpha = 0.7  # values of the alpha parameter for the Gaussian prior

    t, X1, X2 = generate_data(sigma_squared)

    test_mask = (np.abs(Phi[:, 1]) > 0.3) & (np.abs(Phi[:, 2]) > 0.3)
    Phi_train, t_train = Phi[~test_mask], t[~test_mask]
    Phi_test, t_test = Phi[test_mask], t[test_mask]

    # Evaluate the Frequentist model
    mse_freq, w_hat_freq = frequentist_regression(Phi_train, t_train, Phi_test, t_test)
    t_pred_freq = Phi @ w_hat_freq
    print(f'Sigma squared: {sigma_squared}, Frequentist MSE: {mse_freq}')
    plot_data_and_predictions(X1, X2, t, t_pred_freq, "Frequentist", None, sigma_squared)

    # Evaluate the Bayesian model
    t_pred_bayes, posterior_mean, posterior_covariance = bayesian_regression(Phi_train, t_train, alpha, sigma_squared)
    mse_bayes = np.mean((t_test - Phi_test @ posterior_mean) ** 2)
    print(f'Alpha: {alpha}, Bayesian MSE: {mse_bayes}')
    plot_data_and_predictions(X1, X2, t, Phi @ posterior_mean, "Bayesian", alpha, sigma_squared)

if __name__ == "__main__":
    main()

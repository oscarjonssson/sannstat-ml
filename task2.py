
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

# Define the weights
w = np.array([0, 2.5, -0.5])

# Generate the input space grid
x1 = np.arange(-1, 1.05, 0.05)
x2 = np.arange(-1, 1.05, 0.05)
X1, X2 = np.meshgrid(x1, x2)

# Calculate phi(x)
Phi = np.vstack((np.ones(X1.size), X1.ravel()**2, X2.ravel()**3)).T

# Function to calculate outputs for a specific sigma value
def generate_data(sigma):
    epsilon = np.random.normal(0, sigma, size=Phi.shape[0])
    t = np.dot( Phi, w) + epsilon # t = phi(x) * w + epsilon
    return t, X1.ravel(), X2.ravel()

def frequentist_regression(Phi_train, t_train, Phi_test, t_test):
    # Beräkna viktade skattningar och MSE för Frequentist-modellen
    w_hat = inv(Phi_train.T @ Phi_train) @ Phi_train.T @ t_train #formel w_hat = (Phi^T * Phi)^-1 * Phi^T * t 19/21
    t_pred_freq = Phi_test @ w_hat # formeln t = Phi * w_hat, fit the model
    mse_freq = np.mean((t_test - t_pred_freq) ** 2)

    return mse_freq, w_hat

def bayesian_regression(Phi_train, t_train, Phi_test, t_test, alpha, sigma):
    beta = 1.0 / (sigma ** 2)
    # Prior mean and covariance matrix of the weights
    prior_mean = np.zeros(Phi_train.shape[1]) # vad vi tror att vikterna är innan vi har sett någon data
    prior_covariance = alpha * np.eye(Phi_train.shape[1]) 
    # Posterior covariance matrix
    posterior_covariance = inv(inv(prior_covariance) + beta * Phi_train.T @ Phi_train) #formel S_N = (S_0^-1 + beta * Phi^T * Phi)^-1 eller S_N^-1 = (S_0^-1 + beta * Phi^T * Phi) (26/28)
    # Posterior mean(W.T)
    posterior_mean = posterior_covariance @ (inv(prior_covariance) @ prior_mean + beta * Phi_train.T @ t_train ) #formel m_N = S_N * (S_0^-1 * m_0 + beta * Phi^T * t) (25/27)
    # Predict using the posterior mean
    t_pred_bayes = posterior_mean.T @ Phi_test.T # formeln t = Phi * w_hat, fit the model
    # Variance calculations for training and test data
    pred_var_train = [(1 / beta) + np.dot(phi_x.T, np.dot(posterior_covariance, phi_x)) for phi_x in Phi_train] #baseras på formel 31   
    pred_var_test = [(1 / beta) + np.dot(phi_x.T, np.dot(posterior_covariance, phi_x)) for phi_x in Phi_test] #baseras på formel 31
    mse_bayes = np.mean((t_test - t_pred_bayes) ** 2)

    return mse_bayes, pred_var_test, pred_var_train, pred_var_test, posterior_mean


def plot_data_and_predictions(X1, X2, t, t_pred_freq, t_pred_bayes, sigma, alpha):
    fig = plt.figure(figsize=(12, 6))  # Adjusted for better display of two subplots

    # Frequentist predictions
    ax1 = fig.add_subplot(121, projection='3d')  # 121 - 1 row, 2 columns, 1st subplot
    ax1.scatter(X1, X2, t, color='blue', label='Actual Data', alpha=0.6)
    ax1.scatter(X1, X2, t_pred_freq, color='red', label='Frequentist Predictions', alpha=0.6)
    ax1.set_title(f"Frequentist Regression (σ={sigma})")
    ax1.set_xlabel('$x_1$')
    ax1.set_ylabel('$x_2$')
    ax1.set_zlabel('Output $t$')
    ax1.legend()

    # Bayesian predictions
    ax2 = fig.add_subplot(122, projection='3d')  # 122 - 1 row, 2 columns, 2nd subplot
    ax2.scatter(X1, X2, t, color='blue', label='Actual Data', alpha=0.6)
    ax2.scatter(X1, X2, t_pred_bayes, color='green', label='Bayesian Predictions', alpha=0.6)
    ax2.set_title(f"Bayesian Regression (σ={sigma}, α={alpha})")
    ax2.set_xlabel('$x_1$')
    ax2.set_ylabel('$x_2$')
    ax2.set_zlabel('Output $t$')
    ax2.legend()

    plt.tight_layout()
    plt.show()

   
def main():
    sigma_values = [0.2, 0.4, 0.6]  # Different values of sigma 
    alpha_values = [0.7, 1.5, 3.0]  # Different values of alpha for Bayesian regression
    for sigma in sigma_values:
        t, X1, X2 = generate_data(sigma)

        # dela upp data i tränings- och testdata, test data är där både |x1| och |x2| är större än 0.3
        test_mask = (np.abs(Phi[:, 1]) > 0.3) & (np.abs(Phi[:, 2]) > 0.3)
        Phi_train, t_train = Phi[~test_mask], t[~test_mask]
        Phi_test, t_test = Phi[test_mask], t[test_mask]

        mse_freq, w_hat = frequentist_regression(Phi_train, t_train, Phi_test, t_test)
        t_pred_freq = Phi @ w_hat

        alpha_values = [0.7, 1.5, 3.0]
        for alpha in alpha_values:
            mse_bayes, pred_var_test, pred_var_train, pred_var_test, posterior_mean = bayesian_regression(Phi_train, t_train, Phi_test, t_test, alpha, sigma)
            t_pred_bayes = posterior_mean.T @ Phi.T
            print(f'Sigma: {sigma}, Frequentist MSE: {mse_freq},\nSigma: {sigma}, Alpha: {alpha}, Bayesian MSE: {mse_bayes}')
            plot_data_and_predictions(X1, X2, t, t_pred_freq, t_pred_bayes, sigma, alpha)
            print(f"Average Prediction Variance on Training Data: {np.mean(pred_var_train)}")
            print(f"Average Prediction Variance on Test Data: {np.mean(pred_var_test)}\n")

if __name__ == "__main__":
    main()

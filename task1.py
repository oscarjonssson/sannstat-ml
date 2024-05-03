import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Definiera parametrar för prior distribution
alpha = 2  # Eller du kan ändra detta till 1 eller 3 baserat på din uppgiftsinställning
mean = np.array([0, 0])  # Medelvärdet för prior
covariance = np.linalg.inv(alpha * np.eye(2))  # Kovariansmatrisen för prior

# Skapa ett grid för w0 och w1
w0 = np.linspace(-3, 3, 100)
w1 = np.linspace(-3, 3, 100)
W0, W1 = np.meshgrid(w0, w1)
pos = np.dstack((W0, W1))

# Beräkna prior distribution på gridet
prior_rv = multivariate_normal(mean, covariance)
prior_pdf = prior_rv.pdf(pos)

# Plotta prior distribution
plt.figure(figsize=(6, 6))
plt.contourf(W0, W1, prior_pdf, levels=50, cmap='viridis')
plt.title('Prior Distribution over Weights w0 and w1')
plt.xlabel('w0')
plt.ylabel('w1')
plt.colorbar()
plt.show()

# Antag att du har ett datapunkt x och t
x = np.array([1, 0.5])  # x i utökad form, inklusive bias term som 1
t = 0.7  # målvariabeln
beta = 25  # Precision för observationsbrus

# Beräkna likelihood för en datapunkt
def likelihood(w, x, t, beta):
    mu = np.dot(w, x)
    variance = 1 / beta
    return multivariate_normal(mu, variance).pdf(t)

# Skapa en array för likelihood över samma grid
likelihood_pdf = np.array([[likelihood(np.array([w0, w1]), x, t, beta) for w0, w1 in zip(W0_row, W1_row)] for W0_row, W1_row in zip(W0, W1)])

# Plotta likelihood
plt.figure(figsize=(6, 6))
plt.contourf(W0, W1, likelihood_pdf, levels=50, cmap='plasma')
plt.title('Likelihood of t given w0 and w1')
plt.xlabel('w0')
plt.ylabel('w1')
plt.colorbar()
plt.show()

# Beräkna den posteriora distributionen (ignorerar normaliseringskonstanten)
posterior_pdf = prior_pdf * likelihood_pdf

# Plotta posterior distribution
plt.figure(figsize=(6, 6))
plt.contourf(W0, W1, posterior_pdf, levels=50, cmap='coolwarm')
plt.title('Posterior Distribution after observing one data point')
plt.xlabel('w0')
plt.ylabel('w1')
plt.colorbar()
plt.show()

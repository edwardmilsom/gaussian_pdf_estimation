import torch
import numpy as np
from scipy.stats import norm

# Function to calculate Gaussian PDF
def gaussian_pdf(y, mu, sigma):
    return 1 / (torch.sqrt(2 * np.pi * sigma ** 2)) * torch.exp(-(y - mu) ** 2 / (2 * sigma ** 2))

# Parameters
n_samples = 100
n_repeats = 100000
mu = 0
sigma = 1
sigma_noise = torch.tensor(1.0)  # Converted to tensor

# Storage for squared errors
squared_errors_sample_avg = []
squared_errors_empirical = []

# Repeating the experiment
for _ in range(n_repeats):
    # Different y for each repeat
    y = np.random.normal(mu, sigma)
    
    # Generate samples
    samples = torch.normal(mu, sigma, size=(n_samples,))

    # Calculate empirical mean and variance
    empirical_mu = torch.mean(samples)
    empirical_sigma = torch.sqrt(torch.var(samples))

    # Compute expected likelihood using sample average
    sample_avg_pdf = torch.mean(gaussian_pdf(y, samples, sigma_noise))

    # Compute expected likelihood using empirical mean and variance
    empirical_pdf = gaussian_pdf(y, empirical_mu, torch.sqrt(empirical_sigma ** 2 + sigma_noise ** 2))

    # True PDF
    true_pdf = norm.pdf(y, loc=mu, scale=np.sqrt(sigma ** 2 + sigma_noise.item() ** 2))

    # Squared errors
    squared_errors_sample_avg.append((sample_avg_pdf - true_pdf) ** 2)
    squared_errors_empirical.append((empirical_pdf - true_pdf) ** 2)

# Calculate average squared errors
avg_squared_error_sample_avg = np.mean(squared_errors_sample_avg)
avg_squared_error_empirical = np.mean(squared_errors_empirical)

print(f"Average squared error for sample average approach: {avg_squared_error_sample_avg}")
print(f"Average squared error for empirical approach: {avg_squared_error_empirical}")

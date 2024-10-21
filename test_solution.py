import numpy as np

expected_return=[0.1934,0.1575] 
volatility=[0.3025,0.219] 
corr_matrix=[[1,0.35],[0.35,1]] 
risk_free_rate= 0.045 
portfolio_size=2 
risk_aversion=3

cov_matrix = [[0.0 for _ in range(len(volatility))] for _ in range(len(volatility))]

# Calculate the covariance matrix using nested loops
for i in range(len(volatility)):
    for j in range(len(volatility)):
        cov_matrix[i][j] = volatility[i] * volatility[j] * corr_matrix[i][j]

inv_cov_matrix = np.linalg.inv(cov_matrix)


def calculate_intermediate_quantities():
        """Calculate intermediate quantities used in portfolio optimization."""
        u = np.ones(portfolio_size)
        A = sum([sum(u[i] * expected_return[j] * inv_cov_matrix[i, j] for i in range(portfolio_size)) for j in range(portfolio_size)])
        B = sum([sum(expected_return[i] * expected_return[j] * inv_cov_matrix[i, j] for i in range(portfolio_size)) for j in range(portfolio_size)])
        C = sum([sum(u[i] * u[j] * inv_cov_matrix[i, j] for i in range(portfolio_size)) for j in range(portfolio_size)])
        D = B * C - A ** 2
        M = np.dot(np.ones(portfolio_size), inv_cov_matrix)
        L = expected_return @ inv_cov_matrix

        G = (B * M - A * L) / D
        H = (C * L - A * M) / D
        
        return C, G, H
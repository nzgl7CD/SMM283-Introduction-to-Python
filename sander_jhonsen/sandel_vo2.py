# Remember to download the packages 'numpy', 'pandas', 'tkinter', 'messagebox' 'matplotlib', 'jinja2, and 'openpyxl' using the 'pip install' command in CMD  
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox, simpledialog, Entry
import matplotlib.pyplot as plt


# Creating error messages for uncomplete inputs
def is_valid_number_list(input_str):
    try:
        numbers = [float(val) for val in input_str.split()]
        return True, numbers
    except ValueError:
        return False, []


# Calculation of the intermediate values
def calculate_intermediate_values(inv_cov_matrix, expected_returns, unit_vector):
    L = inv_cov_matrix @ expected_returns
    M = inv_cov_matrix @ unit_vector
    A = np.dot(unit_vector, L)
    B = np.dot(expected_returns, L)
    C = np.dot(unit_vector, inv_cov_matrix @ unit_vector)
    D = B * C - A**2
    return L, M, A, B, C, D

def calculate_G_H_and_sum(A, B, C, D, M, L):
    G = (M * B - L * A) / D
    H = (L * C - M * A) / D
    sum_G_H = G + H
    return G, H, sum_G_H


def return_function(a):
    ret = []
    for i in range(1, len(a)):
        ret.append((a[i]-a[i-1])/a[i-1])
    return ret


def compute_portfolio_metrics(C, G, H, target_returns, risk_free_rate, expected_return, cov_matrix, risk_aversion_coefficient, inv_cov_matrix, portfolio_size):
    results = []
    min_var_weights=np.dot(inv_cov_matrix, np.ones(portfolio_size)) /C
    for t in target_returns:
        opt_weights = G + t * H
        mean_var_weight=(1-t)*min_var_weights+t*opt_weights
        mean_var_weight = np.maximum(mean_var_weight, 0) #Making sure no negative weights
        portfolio_return = np.sum(mean_var_weight*expected_return)
        portfolio_volatility = np.sqrt(np.dot(mean_var_weight.T, np.dot(cov_matrix, mean_var_weight)))
        portfolio_utility = portfolio_return - (0.5 * risk_aversion_coefficient * portfolio_volatility**2)
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

        results.append({
            'Target Return': t,
            'Weights': [f"{num:.4f}" for num in mean_var_weight],
            'Expected Return': round(portfolio_return, 4),
            'Volatility': round(portfolio_volatility, 4),
            'Utility': round(portfolio_utility, 4),
            'Sharpe Ratio': round(sharpe_ratio, 4)
        })
    return pd.DataFrame(results)

# calculation for the minimum variance of the portfolio
def min_var_port(A, C):
    min_var_port = A/C
    return min_var_port
# calculation for the optimal variance of the portfolio
def optimal_variance_port(A, C, D, risk_free_rate):
    res = A/C - (D/C**2)/(risk_free_rate-A/C)
    return res
# calculation for the correlation matrix
def calculate_cov_matrix(correlations, std_devs):
    n = len(std_devs)
    cov_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cov_matrix[i, j] = correlations[i][j] * std_devs[i] * std_devs[j]
    return cov_matrix

# Visualisation of the Efficient Frontier
def plot_efficient_frontier(portfolio_metrics):
    plt.figure(figsize=(10, 6))
    minimum_variance_index=portfolio_metrics['Volatility'].idxmin()
    max_sharpe_ratio_index=portfolio_metrics['Sharpe Ratio'].idxmax()
    max_sharpe_value=portfolio_metrics.loc[max_sharpe_ratio_index, 'Sharpe Ratio']
    min_volatility = portfolio_metrics.loc[minimum_variance_index, 'Volatility']
    min_volatility_return=portfolio_metrics.loc[minimum_variance_index, 'Expected Return']
    plt.plot(portfolio_metrics['Volatility'], portfolio_metrics['Expected Return'], label='Efficient Frontier')
    plt.scatter(min_volatility, min_volatility_return, color='r', marker='o', s=100, zorder=5, label=f'Min Variance Stdv: {min_volatility:.4f}')
    plt.scatter(portfolio_metrics.loc[max_sharpe_ratio_index, 'Volatility'], portfolio_metrics.loc[max_sharpe_ratio_index, 'Expected Return'], color='g', marker='o', s=100, zorder=5, label=f'Max Sharpe Ratio: {max_sharpe_value:.4f}')
    plt.xlabel('Volatility')
    plt.ylabel('Expected Return')
    plt.title('Mean Variance Efficient Frontier')
    plt.legend()
    plt.grid(True)
    plt.show()


def submit(num_assets_entry, expected_returns_input_entry, std_devs_entry, correlations_entry, risk_free_rate_entry, risk_aversion_coefficient_entry):
    num_assets = int(num_assets_entry.get())
    expected_returns_input = list(map(float, expected_returns_input_entry.get().split()))
    std_devs = list(map(float, std_devs_entry.get().split()))
    risk_free_rate = float(risk_free_rate_entry.get())
    risk_aversion_coefficient = float(risk_aversion_coefficient_entry.get())
    
    correlations_input = correlations_entry.get().split()
    correlations = np.array(correlations_input, dtype=float).reshape(num_assets, num_assets)

    main_code(num_assets, expected_returns_input, std_devs, correlations, risk_free_rate, risk_aversion_coefficient)


# Defining the Main Function
def main():
    window = tk.Tk()
    window.title("Portfolio Input")

    tk.Label(window, text="Number of Assets:").grid(row=0, column=0)
    num_assets_entry = Entry(window)
    num_assets_entry.grid(row=0, column=1)

    tk.Label(window, text="Expected Returns (space-separated, Example: 0.04 0.05):").grid(row=1, column=0)
    expected_returns_input_entry = Entry(window)
    expected_returns_input_entry.grid(row=1, column=1)

    tk.Label(window, text="Standard Deviations (space-separated, Example: 0.02 0.03):").grid(row=2, column=0)
    std_devs_entry = Entry(window)
    std_devs_entry.grid(row=2, column=1)

    tk.Label(window, text="Correlations (space-separated, row-wise, Example: 1 0.5 0.5 1):").grid(row=3, column=0)
    correlations_entry = Entry(window)
    correlations_entry.grid(row=3, column=1)

    tk.Label(window, text="Risk Aversion Coefficient (Example: 1.5):").grid(row=4, column=0)
    risk_aversion_coefficient_entry = Entry(window)
    risk_aversion_coefficient_entry.grid(row=4, column=1)

    tk.Label(window, text="Risk Free Rate (Example: 0.01):").grid(row=5, column=0)
    risk_free_rate_entry = Entry(window)
    risk_free_rate_entry.grid(row=5, column=1)

    submit_button = tk.Button(window, text="Submit", command=lambda: submit(num_assets_entry, expected_returns_input_entry, std_devs_entry, correlations_entry, risk_free_rate_entry, risk_aversion_coefficient_entry))
    submit_button.grid(row=6, column=0, columnspan=2)

    window.mainloop()

def main_code(num_assets, expected_returns_input, std_devs, correlations, risk_free_rate, risk_aversion_coefficient):
    unit_vector = np.ones(len(expected_returns_input))
    
    cov_matrix_input = calculate_cov_matrix(correlations, std_devs)
    inv_cov_matrix = np.linalg.inv(cov_matrix_input)

    L, M, A, B, C, D = calculate_intermediate_values(inv_cov_matrix, expected_returns_input, unit_vector)
    print("L:", L)
    print("M:", M)
    print("A:", A)
    print("B:", B)
    print("C:", C)
    print("D:", D)

    G, H, sum_G_and_H = calculate_G_H_and_sum(A, B, C, D, M, L)
    G_return = return_function(G)
    H_return = return_function(H)
    G_and_H_return = return_function(sum_G_and_H)

    print("G:", G)
    print("H:", H)
    print("G+H:", sum_G_and_H)
    print("G return:", G_return)
    print("H return:", H_return)
    print("G+H return:", G_and_H_return)
    print("G risk:", G_risk)
    print("H risk:", H_risk)
    print("G+H risk:", G_and_H_risk)

    min_var_port_return = min_var_port(A, C)
    optimal_variance_port_return = optimal_variance_port(A, C, D, risk_free_rate)
    print("Minimum Variance Portfolio Return:", min_var_port_return)
    print("Optimum Variance Portfolio Return:", optimal_variance_port_return)

    target_returns = np.arange(0.01, 1.01, 0.01)
    portfolio_metrics = compute_portfolio_metrics(C, G, H, target_returns, risk_free_rate, expected_returns_input, cov_matrix_input, risk_aversion_coefficient, inv_cov_matrix, num_assets)
    
    plot_efficient_frontier(portfolio_metrics)

if __name__ == "__main__":
    main()
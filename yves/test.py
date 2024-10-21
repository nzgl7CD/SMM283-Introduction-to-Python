import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


expected_returns = [0.19,0.1575]
expected_returns=np.array(expected_returns)
volatilities = [0.3025,0.2145]
volatilities=np.array(volatilities)
correlation_matrix = [[1,0.35],[0.35,1]]


def calculate_covariance_matrix(volatilities, correlation_matrix):
    num_assets = len(volatilities)
    covariance_matrix = np.zeros((num_assets, num_assets))
    for i in range(num_assets):
        for j in range(num_assets):
            covariance_matrix[i, j] = volatilities[i] * volatilities[j] * correlation_matrix[i][j]
    return covariance_matrix


cov=calculate_covariance_matrix(volatilities, correlation_matrix)


def calculate_hl(num_assets=2, expected_returns=[0.19,0.01], volatilities=[0.3025,0.01], correlation_matrix=[[1,0.35],[0.35,1]], risk_free_rate=0.045):
    covariance_matrix = calculate_covariance_matrix(volatilities, correlation_matrix)
    inverse_covariance_matrix = np.linalg.inv(covariance_matrix)

    ones = np.ones(num_assets)
    expected_returns = np.array(expected_returns)

    A = ones @ inverse_covariance_matrix @ expected_returns
    C = ones @ inverse_covariance_matrix @ ones
    B = expected_returns @ inverse_covariance_matrix @ expected_returns
    D = B * C - A**2

    M = inverse_covariance_matrix @ expected_returns
    L = inverse_covariance_matrix @ ones

    H = (M * C - L * A) / D
    G = (L * B - M * A) / D

    min_var_portfolio = A / C
    opt_var_portfolio = ((A / C) - (D / (C**2)) / (risk_free_rate - A / C))

    print("\nCalculated Values:")
    print("A: {:.4f}".format(A))
    print("B: {:.4f}".format(B))
    print("C: {:.4f}".format(C))
    print("D: {:.4f}".format(D))
    print("M: {}".format(["{:.4f}".format(m) for m in M]))
    print("L: {}".format(["{:.4f}".format(l) for l in L]))
    print("G: {}".format(["{:.4f}".format(g) for g in G]))
    print("H: {}".format(["{:.4f}".format(h) for h in H]))

    return C,G, H, min_var_portfolio, opt_var_portfolio, covariance_matrix

def calculate_portfolio_metrics(G, H, covariance_matrix, expected_returns, risk_aversion, risk_free_rate, min_var_weights):
    target_returns = [i * 0.01 for i in range(101)]
    risks = []
    rets = []
    utils = []
    sharpe_ratios = []
    weights_list = []
    target_returns = [i * 0.01 for i in range(101)]
    for target_return in target_returns:
        opt_var_weights=G+H*target_return
        weights = (1 - target_return) * min_var_weights + target_return * opt_var_weights
        portfolio_return = np.sum(weights * expected_returns)  # Compute portfolio return
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix,weights)))  # Compute portfolio risk
        utility = portfolio_return - 0.5 * risk_aversion * portfolio_risk**2  # Compute utility
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk  # Compute Sharpe ratio

        risks.append(portfolio_risk)  # Ensure it's a scalar
        rets.append(float(portfolio_return))  # Ensure it's a scalar
        utils.append(float(utility))  # Ensure it's a scalar
        sharpe_ratios.append(float(sharpe_ratio))  # Ensure it's a scalar
        weights_list.append(weights)  # Store the weights vector

    return np.array(risks), np.array(rets), np.array(utils), np.array(sharpe_ratios), np.array(weights_list)

def plot_efficient_frontier_and_cml(portolfio_metrics):
    portfolio_risks=[risk for risk in portolfio_metrics[0]]
    portfolio_returns=[ret for ret in portolfio_metrics[1]]
    portfolio_sharpes=[sr for sr in portolfio_metrics[3]]
    
    plt.figure(figsize=(10, 6))

    # Plot the mean-variance efficient frontier
    plt.plot(portfolio_risks, portfolio_returns, marker='o', linestyle='-', color='b', label='Efficient Frontier')

    # Plot the Capital Market Line (CML)
    max_sharpe_index = np.argmax(portfolio_sharpes)
    min_var_point = np.argmin(portfolio_risks)
    
    plt.scatter(portfolio_risks[max_sharpe_index], portfolio_returns[max_sharpe_index], color='red', marker='o', s=100 , zorder=5, label=f'Max Sharpe Ratio: {portfolio_sharpes[max_sharpe_index]:.2f}')
    plt.scatter(portfolio_risks[min_var_point], portfolio_returns[min_var_point], color='green', marker='o', s=100 , zorder=5, label=f'Min Var Point: {portfolio_risks[min_var_point]:.2f}')

    # cml_x = [0, portfolio_risks[max_sharpe_index]]
    # cml_y = [risk_free_rate * 100, portfolio_returns[max_sharpe_index]]
    # plt.plot(cml_x, cml_y, linestyle='--', color='r', label='Capital Market Line (CML)')

    # Add labels and title
    plt.xlabel('Risk (Standard Deviation) %')
    plt.ylabel('Return %')
    plt.title('Mean-Variance Efficient Frontier and CML')
    plt.legend()
    plt.grid(True)
    plt.show()
cov=calculate_covariance_matrix(volatilities, correlation_matrix)
inverse_covariance_matrix = np.linalg.inv(cov)
C, G, H, min_var_portfolio, opt_var_portfolio, _=calculate_hl()
min_var_weight=np.dot(inverse_covariance_matrix, np.ones(2)) /C
metrics=calculate_portfolio_metrics(G,H,cov, expected_returns, 3,0.045,min_var_weight)


plot_efficient_frontier_and_cml(metrics)




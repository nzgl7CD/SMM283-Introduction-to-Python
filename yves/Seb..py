import math
import numpy as np
import matplotlib.pyplot as plt

def get_user_input():
    valid_input = False
    while not valid_input:
        try:
            num_assets = int(input("Enter the number of assets (2-12): "))
            if 2 <= num_assets <= 12:
                valid_input = True
            else:
                print("Please enter a number between 2 and 12.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

    expected_returns = []
    volatilities = []
    correlation_matrix = []

    for i in range(num_assets):
        valid_input = False
        while not valid_input:
            try:
                ret = float(input(f"Enter the expected return for Asset {i+1} (as a decimal): "))
                expected_returns.append(ret)
                valid_input = True
            except ValueError:
                print("Invalid input. Please enter a number.")

        valid_input = False
        while not valid_input:
            try:
                vol = float(input(f"Enter the standard deviation (volatility) for Asset {i+1} (as a decimal): "))
                volatilities.append(vol)
                valid_input = True
            except ValueError:
                print("Invalid input. Please enter a number.")
                
    
    print("Enter the correlation matrix (values between -1 and 1):")
    for i in range(num_assets):
        row = []
        for j in range(num_assets):
            valid_input = False
            while not valid_input:
                try:
                    if i == j:
                        row.append(1.0)
                        valid_input = True
                    else:
                        corr = float(input(f"Correlation between Asset {i+1} and Asset {j+1}: "))
                        if -1 <= corr <= 1:
                            row.append(corr)
                            valid_input = True
                        else:
                            print("Please enter a value between -1 and 1.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
        correlation_matrix.append(row)

    valid_input = False
    while not valid_input:
        try:
            risk_aversion = float(input("Enter the risk aversion coefficient: "))
            valid_input = True
        except ValueError:
            print("Invalid input. Please enter a number.")

    valid_input = False
    while not valid_input:
        try:
            risk_free_rate = float(input("Enter the risk-free rate (as a decimal): "))
            valid_input = True
        except ValueError:
            print("Invalid input. Please enter a number.")

    return num_assets, expected_returns, volatilities, correlation_matrix, risk_aversion, risk_free_rate

def calculate_covariance_matrix(volatilities, correlation_matrix):
    num_assets = len(volatilities)
    covariance_matrix = np.zeros((num_assets, num_assets))
    for i in range(num_assets):
        for j in range(num_assets):
            covariance_matrix[i, j] = volatilities[i] * volatilities[j] * correlation_matrix[i][j]
    return covariance_matrix

def calculate_hl(num_assets, expected_returns, volatilities, covariance_matrix, risk_free_rate):
    # covariance_matrix = calculate_covariance_matrix(volatilities, correlation_matrix)
    inverse_covariance_matrix = np.linalg.inv(covariance_matrix)
    ones = np.ones(num_assets)

    A = ones @ inverse_covariance_matrix @ expected_returns
    C = ones @ inverse_covariance_matrix @ ones
    B = expected_returns @ inverse_covariance_matrix @ expected_returns
    D = B * C - A**2

    M = inverse_covariance_matrix @ expected_returns
    L = inverse_covariance_matrix @ ones

    H = (M * C - L * A) / D
    G = (L * B - M * A) / D

    # min_var_portfolio = A / C
    # opt_var_portfolio = ((A / C) - (D / (C**2)) / (risk_free_rate - A / C))

    print("\nCalculated Values:")
    print("A: {:.4f}".format(A))
    print("B: {:.4f}".format(B))
    print("C: {:.4f}".format(C))
    print("D: {:.4f}".format(D))
    print("M: {}".format(["{:.4f}".format(m) for m in M]))
    print("L: {}".format(["{:.4f}".format(l) for l in L]))
    print("G: {}".format(["{:.4f}".format(g) for g in G]))
    print("H: {}".format(["{:.4f}".format(h) for h in H]))

    return C, G, H, covariance_matrix, inverse_covariance_matrix

def calculate_portfolio_metrics(C, G, H, covariance_matrix,inverse_covariance_matrix, expected_returns, risk_aversion, risk_free_rate, portfolio_size):
    target_returns = [i * 0.01 for i in range(101)]
    risks = []
    rets = []
    utils = []
    sharpe_ratios = []
    weights_list = []
    
    min_var_weights= np.dot(inverse_covariance_matrix, np.ones(portfolio_size)) /C

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

    return np.array(risks), np.array(rets), np.array(utils), np.array(sharpe_ratios), np.array(weights_list), target_returns

def print_covariance_matrix(covariance_matrix):
    print("\nCovariance Matrix:")
    for row in covariance_matrix:
        print(" ".join("{:8.4f}".format(x) for x in row))

def plot_efficient_frontier_and_cml(portfolio_risks, portfolio_returns, risk_free_rate, sharpe_ratios):
    plt.figure(figsize=(10, 6))

    # Plot the mean-variance efficient frontier
    plt.plot(portfolio_risks, portfolio_returns, marker='o', linestyle='-', color='b', label='Efficient Frontier')

    # Plot the Capital Market Line (CML)
    max_sharpe_index = np.argmax(sharpe_ratios)
    cml_x = [0, portfolio_risks[max_sharpe_index]]
    cml_y = [risk_free_rate * 100, portfolio_returns[max_sharpe_index]]
    plt.plot(cml_x, cml_y, linestyle='--', color='r', label='Capital Market Line (CML)')

    # Add labels and title
    plt.xlabel('Risk (Standard Deviation) %')
    plt.ylabel('Return %')
    plt.title('Mean-Variance Efficient Frontier and CML')
    plt.legend()
    plt.grid(True)
    plt.show()

def print_results(target_returns, portfolio_risks, portfolio_returns, utilities, sharpe_ratios, weights_list):
    print("\nResults:")
    print("{:<5s} {:>12s} {:>12s} {:>12s} {:>18s}".format("No#", "Return (%)", "Risk (%)", "Utility", "Sharpe Ratio (%)"))
    for i, (ret, risk, utility, sharpe_ratio, weights) in enumerate(zip(portfolio_returns, portfolio_risks, utilities, sharpe_ratios, weights_list)):
        # Debugging prints
        print(f"ret: {ret}, type: {type(ret)}")
        print(f"risk: {risk}, type: {type(risk)}")
        print(f"utility: {utility}, type: {type(utility)}")
        print(f"sharpe_ratio: {sharpe_ratio}, type: {type(sharpe_ratio)}")

        print("{:<5d} {:>12.4f} {:>12.4f} {:>12.4f} {:>18.4f}".format(i+1, float(ret), float(risk), float(utility), float(sharpe_ratio)))
        weights_str = " ".join(["{:.4f}".format(x) for x in weights])
        print(f"Weights: {weights_str}")
        

def plot_efficient_frontier_and_cml(portolfio_metrics):
    portfolio_risks=[risk for risk in portolfio_metrics[0]]
    portfolio_returns=[ret for ret in portolfio_metrics[1]]
    portfolio_sharpes=[sr for sr in portolfio_metrics[3]]
    plt.figure(figsize=(10, 6))
    plt.plot(portfolio_risks, portfolio_returns, marker='o', linestyle='-', color='b', label='Efficient Frontier')
    max_sharpe_index = np.argmax(portfolio_sharpes)
    min_var_point = np.argmin(portfolio_risks)
    plt.scatter(portfolio_risks[max_sharpe_index], portfolio_returns[max_sharpe_index], color='yellow', marker='o', s=100 , zorder=5, label=f'Max Sharpe Ratio: {portfolio_sharpes[max_sharpe_index]:.5f}')
    plt.scatter(portfolio_risks[min_var_point], portfolio_returns[min_var_point], color='red', marker='o', s=100 , zorder=5, label=f'Min Var Point: {portfolio_risks[min_var_point]:.5f}')
    plt.xlabel('Standard Deviation')
    plt.ylabel('Return')
    plt.title('Efficient Frontier')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    num_assets, expected_returns, volatilities, correlation_matrix, risk_aversion_coefficient, risk_free_rate = get_user_input()
    
    expected_returns=np.array(expected_returns)
    volatilities=np.array(volatilities)
    covariance_matrix=calculate_covariance_matrix(volatilities, correlation_matrix)
    
    C, G, H, covariance_matrix, inverse_covariance_matrix = calculate_hl(num_assets, expected_returns, volatilities, covariance_matrix, risk_free_rate)
    
    metrics = calculate_portfolio_metrics(C, G, H, covariance_matrix,inverse_covariance_matrix, expected_returns, risk_aversion_coefficient, risk_free_rate,num_assets)
    plot_efficient_frontier_and_cml(metrics)
    portfolio_risks, portfolio_returns, utilities, sharpe_ratios, weights_list,target_returns=metrics

    print_covariance_matrix(covariance_matrix)
    print_results(target_returns, portfolio_risks, portfolio_returns, utilities, sharpe_ratios, weights_list)

    print("\nPortfolio with the Highest Utility:")
    max_utility_index = np.argmax(utilities)
    print("Return: {:.4f}, Risk: {:.4f}, Utility: {:.4f}, Sharpe Ratio: {:.4f}".format(
        float(portfolio_returns[max_utility_index]), float(portfolio_risks[max_utility_index]), float(utilities[max_utility_index]), float(sharpe_ratios[max_utility_index])))

    print("\nPortfolio with the Highest Sharpe Ratio:")
    max_sharpe_index = np.argmax(sharpe_ratios)
    print("Return: {:.4f}%, Risk: {:.4f}%, Utility: {:.4f}, Sharpe Ratio: {:.4f}".format(
        float(portfolio_returns[max_sharpe_index]), float(portfolio_risks[max_sharpe_index]), float(utilities[max_sharpe_index]), float(sharpe_ratios[max_sharpe_index])))

    

if __name__ == "__main__":
    main()

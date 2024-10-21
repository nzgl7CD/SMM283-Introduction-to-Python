import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class PortfolioOptimizer:
    def __init__(self, expected_return=None, volatility=None, corr_matrix=None, risk_free_rate=0.045, portfolio_size=2, risk_aversion=3.0):
        self.risk_free_rate = risk_free_rate
        self.portfolio_size = portfolio_size
        self.risk_aversion = risk_aversion
        self.dataframe = None

        if expected_return is not None and volatility is not None and corr_matrix is not None:
            self.returns = np.asarray(expected_return)
            self.cov_matrix = self.calculate_covariance_matrix(volatility, corr_matrix)
        else:
            self.ds = self.get_data(portfolio_size)
            self.returns = self.calculate_annualized_returns()
            self.cov_matrix = self.compute_covariance_matrix(self.ds)

        self.inv_cov_matrix = np.linalg.inv(self.cov_matrix)
        self.C, self.G, self.H = self.calculate_intermediate_quantities()

    def calculate_covariance_matrix(self, volatility, corr_matrix):
        corr_matrix = np.array(corr_matrix)
        stdv = np.array(volatility)
        return np.outer(stdv, stdv) * corr_matrix

    def get_data(self, n):
        ds = pd.read_excel('230023476PortfolioProblem.xlsx')
        ds['Date'] = pd.to_datetime(ds['Date'])
        ds.iloc[:, 1:] = ds.iloc[:, 1:].pct_change()
        return ds.iloc[:, :n + 1].dropna()

    def calculate_annualized_returns(self):
        returns = self.ds.iloc[:, 1:]
        compounded_returns = (returns + 1).prod() ** (12 / len(returns)) - 1
        return compounded_returns.values

    def compute_covariance_matrix(self, dataset):
        return dataset.drop(columns=['Date']).cov() * 12

    def calculate_intermediate_quantities(self):
        u = np.ones(self.portfolio_size)
        inv_cov_matrix = self.inv_cov_matrix
        A = np.dot(u, np.dot(inv_cov_matrix, self.returns))
        B = np.dot(self.returns, np.dot(inv_cov_matrix, self.returns))
        C = np.dot(u, np.dot(inv_cov_matrix, u))
        D = B * C - A ** 2
        G = (np.dot(inv_cov_matrix, u) * B - np.dot(inv_cov_matrix, self.returns) * A) / D
        H = (np.dot(inv_cov_matrix, self.returns) * C - np.dot(inv_cov_matrix, u) * A) / D
        return C, G, H

    def calculate_portfolio_metrics(self, weights):
        portfolio_return = np.sum(weights * self.returns)
        portfolio_variance = np.dot(weights.T, np.dot(self.cov_matrix, weights))
        portfolio_risk = np.sqrt(portfolio_variance)
        excess_return = portfolio_return - self.risk_free_rate
        sharpe_ratio = excess_return / portfolio_risk
        utility = portfolio_return - (0.5 * self.risk_aversion * portfolio_variance)
        return portfolio_return, portfolio_risk, sharpe_ratio, utility

    def calculate_minimum_variance_weights(self):
        return np.dot(self.inv_cov_matrix, np.ones(self.portfolio_size)) / self.C

    def calculate_optimum_variance_weights(self, target_return):
        return self.G + (target_return * self.H)

    def calculate_mean_variance_efficient_frontier(self):
        min_var_weights = self.calculate_minimum_variance_weights()
        frontier_weights = [(1 - target_return) * min_var_weights + target_return * self.calculate_optimum_variance_weights(target_return) for target_return in np.linspace(0, 1, 101)]
        frontier_metrics = [self.calculate_portfolio_metrics(w) for w in frontier_weights]
        return frontier_weights, frontier_metrics

    def plot_efficient_frontier(self, ax):
        _, frontier_metrics = self.calculate_mean_variance_efficient_frontier()
        frontier_risks = [metric[1] for metric in frontier_metrics]
        frontier_returns = [metric[0] for metric in frontier_metrics]
        sharpe_ratios = [metric[2] for metric in frontier_metrics]

        min_var_idx = np.argmin(frontier_risks)
        min_var_point = frontier_metrics[min_var_idx]

        max_sharpe_idx = np.argmax(sharpe_ratios)
        max_sharpe_point = frontier_metrics[max_sharpe_idx]

        ax.plot(frontier_risks, frontier_returns, 'b-o', label='Efficient Frontier')
        ax.scatter(min_var_point[1], min_var_point[0], color='green', marker='o', s=100, zorder=5, label=f'Min Variance Stdv: {min_var_point[1]:.4f}')
        ax.scatter(max_sharpe_point[1], max_sharpe_point[0], color='red', marker='o', s=100, zorder=5, label=f'Max Sharpe Ratio: {max_sharpe_point[2]:.4f}')
        ax.set_xlabel('Portfolio Volatility (Risk)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Portfolio Return', fontsize=12, fontweight='bold')
        ax.set_title('Mean-Variance Efficient Frontier', fontsize=14, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=10)
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax.axvline(0, color='black', linewidth=0.5, linestyle='--')
        ax.set_xlim(0, max(frontier_risks) + 0.05)

    def write_to_excel(self, output_file='230023476PortfolioProblem.xlsx'):
        frontier_weights, frontier_metrics = self.calculate_mean_variance_efficient_frontier()

        if hasattr(self, 'ds'):
            weight_columns = [f'w_{col}' for col in self.ds.columns[1:]]
        else:
            weight_columns = [f'w{i+1}' for i in range(self.portfolio_size)]

        data = {
            'Return': [metric[0] for metric in frontier_metrics],
            'Volatility': [metric[1] for metric in frontier_metrics],
            'Sharpe Ratio': [metric[2] for metric in frontier_metrics],
            'Utility': [metric[3] for metric in frontier_metrics]
        }

        for i, col in enumerate(weight_columns):
            data[col] = [w[i] for w in frontier_weights]

        df = pd.DataFrame(data)
        df.sort_values(by='Return', inplace=True)
        df = df.round(4)

        with pd.ExcelWriter(output_file, mode='a', engine="openpyxl", if_sheet_exists="replace") as writer:
            df.to_excel(writer, sheet_name='output', index=False)
            workbook = writer.book
            worksheet = workbook['output']

            for column_cells in worksheet.columns:
                max_length = 0
                column = column_cells[0].column_letter
                for cell in column_cells:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(cell.value)
                    except:
                        pass
                adjusted_width = (max_length + 2) * 1.2
                worksheet.column_dimensions[column].width = adjusted_width

            self.dataframe = df

    def print_values(self):
        try:
            if self.dataframe is None:
                raise ValueError("Dataframe is not initialized. Please ensure the calculations are performed before calling this method.")

            df = self.dataframe

            if df.empty:
                raise ValueError("Dataframe is empty. Please check the input data and calculations.")

            max_sharpe_idx = df['Sharpe Ratio'].idxmax()
            max_sharpe_return, max_sharpe_volatility, max_sharpe_value, max_sharpe_utility = df.loc[max_sharpe_idx, ['Return', 'Volatility', 'Sharpe Ratio', 'Utility']]

            max_utility_idx = df['Utility'].idxmax()
            max_utility_return, max_utility_volatility, max_utility_sharpe, max_utility_value = df.loc[max_utility_idx, ['Return', 'Volatility', 'Sharpe Ratio', 'Utility']]

            min_volatility_idx = df['Volatility'].idxmin()
            min_volatility_return, min_volatility_volatility, min_volatility_sharpe, min_volatility_utility = df.loc[min_volatility_idx, ['Return', 'Volatility', 'Sharpe Ratio', 'Utility']]

            return {
                'MaxSharpeRatio': [max_sharpe_idx, max_sharpe_return, max_sharpe_volatility, max_sharpe_value, max_sharpe_utility],
                'MaxUtility': [max_utility_idx, max_utility_return, max_utility_volatility, max_utility_sharpe, max_utility_value],
                'MinVolatility': [min_volatility_idx, min_volatility_return, min_volatility_volatility, min_volatility_sharpe, min_volatility_utility]
            }

        except Exception as e:
            raise ValueError(f"An error occurred while printing values: {str(e)}")

class PortfolioApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mean-Variance Portfolio Optimizer")
        self.create_widgets()

    def create_widgets(self):
        self.frame = ttk.Frame(self.root, padding="10")
        self.frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.label_risk_aversion = ttk.Label(self.frame, text="Risk Aversion:", font=('Helvetica', 12))
        self.label_risk_aversion.grid(row=0, column=0, sticky=tk.W, pady=5)

        self.entry_risk_aversion = ttk.Entry(self.frame, font=('Helvetica', 12))
        self.entry_risk_aversion.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5)
        self.entry_risk_aversion.insert(0, "3.0")

        self.button_calculate = ttk.Button(self.frame, text="Calculate", command=self.calculate)
        self.button_calculate.grid(row=1, column=0, columnspan=2, pady=10)

        self.label_results = ttk.Label(self.frame, text="", font=('Helvetica', 12))
        self.label_results.grid(row=2, column=0, columnspan=2, pady=5)

        self.figure = plt.Figure(figsize=(10, 6), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, self.frame)
        self.canvas.get_tk_widget().grid(row=3, column=0, columnspan=2, pady=10)

    def calculate(self):
        risk_aversion = float(self.entry_risk_aversion.get())
        optimizer = PortfolioOptimizer(risk_aversion=risk_aversion)
        optimizer.write_to_excel()
        optimizer.plot_efficient_frontier(self.ax)
        self.canvas.draw()

        results = optimizer.print_values()
        self.label_results['text'] = (
            f"Max Sharpe Ratio: {results['MaxSharpeRatio'][3]:.4f}\n"
            f"Max Utility: {results['MaxUtility'][4]:.4f}\n"
            f"Min Volatility: {results['MinVolatility'][2]:.4f}"
        )

if __name__ == "__main__":
    root = tk.Tk()
    app = PortfolioApp(root)
    root.mainloop()

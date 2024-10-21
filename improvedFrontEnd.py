"""
Author: sebastianveum
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class OptimalPortfolio:
    def __init__(self, portfolio_size, risk_aversion, risk_free_rate, expected_returns, covariance_matrix):
        
        self._portfolio_size = portfolio_size
        self._risk_aversion = risk_aversion
        self._risk_free_rate = risk_free_rate
        self._expected_returns = expected_returns
        self._covariance_matrix = covariance_matrix
        self._inv_cov_matrix = np.linalg.inv(covariance_matrix)
        self._C, self._G, self._H = self._calculate_intermediate_quantities()
    
    def _calculate_intermediate_quantities(self):
        """Calculate intermediate quantities used in portfolio optimization."""
        u = np.ones(self._portfolio_size)
        A = sum([sum(u[i] * self._expected_returns[j] * self._inv_cov_matrix[i, j] for i in range(self._portfolio_size)) for j in range(self._portfolio_size)])
        B = sum([sum(self._expected_returns[i] * self._expected_returns[j] * self._inv_cov_matrix[i, j] for i in range(self._portfolio_size)) for j in range(self._portfolio_size)])
        C = sum([sum(u[i] * u[j] * self._inv_cov_matrix[i, j] for i in range(self._portfolio_size)) for j in range(self._portfolio_size)])
        D = B * C - A ** 2
        M = np.dot(np.ones(self._portfolio_size), self._inv_cov_matrix)
        L = self._expected_returns @ self._inv_cov_matrix

        G = (B * M - A * L) / D
        H = (C * L - A * M) / D
        
        return C, G, H
    
    def calculate_portfolio_metrics(self, weights):
        """Calculate return, risk, Sharpe ratio, and utility of the portfolio."""
        portfolio_return = np.sum(weights * self._expected_returns)
        portfolio_variance = np.dot(weights.T, np.dot(self._covariance_matrix, weights))
        portfolio_risk = np.sqrt(portfolio_variance)
        excess_return = portfolio_return - self._risk_free_rate
        sharpe_ratio = excess_return / portfolio_risk
        utility = portfolio_return - (0.5 * self._risk_aversion * portfolio_variance)
        return portfolio_return, portfolio_risk, sharpe_ratio, utility

    def calculate_minimum_variance_weights(self):
        """Calculate weights for the minimum variance portfolio."""
        return np.dot(self._inv_cov_matrix, np.ones(self._portfolio_size)) / self._C
        
    def calculate_optimum_variance_weights(self, target_return):
        """Calculate weights for the optimal variance portfolio for a given target return."""
        return self._G + (target_return * self._H)

    def calculate_mean_variance_efficient_frontier(self):
        """Calculate the efficient frontier values for mean-variance optimization from 0 to 100% target return"""
        min_var_weights = self.calculate_minimum_variance_weights()
        frontier_weights = []
        for target_return in np.linspace(0, 1, 101):
            opt_var_weights = self.calculate_optimum_variance_weights(target_return)
            weights = (1 - target_return) * min_var_weights + target_return * opt_var_weights
            frontier_weights.append(weights)
        frontier_metrics = [self.calculate_portfolio_metrics(w) for w in frontier_weights]
        return frontier_weights, frontier_metrics

class PortfolioVisualising:
    def __init__(self, root):
        """Initialize the frontend interface"""
        self.root = root
        self.root.title("Portfolio Optimizer")
        self.style = ttk.Style()
        self.style.configure('TButton', font=('Inter', 12), padding=8, background='#4CAF50', foreground='#000000')
        self.style.map('TButton', background=[('active', '#45a049')])
        self._create_widgets_interface()
        self._values_for_print = None
        
    def _create_widgets_interface(self):
        """Create and arrange all the widgets in the application"""
        self.frame = ttk.Frame(self.root, padding="20")
        self.frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Portfolio Size
        ttk.Label(self.frame, text="Portfolio Size (2-12 securities):").grid(column=0, row=0, padx=10, pady=10, sticky=tk.W)
        self.portfolio_size_entry = ttk.Entry(self.frame, width=10, font=('Inter', 12))
        self.portfolio_size_entry.grid(column=1, row=0, padx=10, pady=10)
        self.portfolio_size_entry.insert(tk.END, "2")
        
        # Risk-Free Rate
        ttk.Label(self.frame, text="Risk-Free Rate:").grid(column=0, row=2, padx=10, pady=10, sticky=tk.W)
        self.risk_free_rate_entry = ttk.Entry(self.frame, width=10, font=('Inter', 12))
        self.risk_free_rate_entry.grid(column=1, row=2, padx=10, pady=10)
        self.risk_free_rate_entry.insert(tk.END, "0.045")

        # Risk Aversion
        ttk.Label(self.frame, text="Risk Aversion:").grid(column=0, row=1, padx=10, pady=10, sticky=tk.W)
        self.risk_aversion_entry = ttk.Entry(self.frame, width=10, font=('Inter', 12))
        self.risk_aversion_entry.grid(column=1, row=1, padx=10, pady=10)
        self.risk_aversion_entry.insert(tk.END, "3.0")

        # Expected Returns
        ttk.Label(self.frame, text="Expected Returns (comma-separated):").grid(column=0, row=3, padx=10, pady=10, sticky=tk.W)
        self.expected_returns_text = tk.Text(self.frame, width=60, height=1, font=('Inter', 12))
        self.expected_returns_text.grid(column=1, row=3, columnspan=2, padx=10, pady=10)
        self.expected_returns_text.insert(tk.END, "0.1934, 0.1575") # Suggest standards from excel file provided
        
        # Volatilities
        ttk.Label(self.frame, text="Volatilities (comma-separated):").grid(column=0, row=4, padx=10, pady=10, sticky=tk.W)
        self.volatilities_text = tk.Text(self.frame, width=60, height=1, font=('Inter', 12))
        self.volatilities_text.grid(column=1, row=4, columnspan=2, padx=10, pady=10)
        self.volatilities_text.insert(tk.END, "0.3025, 0.219") # Suggest standards from excel file provided

        # Correlation Matrix (adjusted as suggested by Adrian)
        ttk.Label(self.frame, text="Correlation Matrix:").grid(column=0, row=5, padx=10, pady=10, sticky=tk.W)
        self.correlation_matrix_var = tk.StringVar()
        self.correlation_matrix_var.set("1, 0.35\n0.35, 1")
        
        self.correlation_matrix_label = ttk.Label(self.frame, textvariable=self.correlation_matrix_var, wraplength=400, justify=tk.LEFT, font=('Inter', 10))
        self.correlation_matrix_label.grid(column=2, row=5, columnspan=2, padx=10, pady=10, sticky=tk.W)
        
        # Edit Correlation Matrix Button
        self.correlation_matrix_button = ttk.Button(self.frame, text='Edit Correlation Matrix', command=self.render_correlation_matrix)
        self.correlation_matrix_button.grid(column=1, row=5, padx=10, pady=10)

        # Run Optimization Button
        self.optimize_button = ttk.Button(self.frame, text="Run Optimization", command=self.optimize_portfolio)
        self.optimize_button.grid(column=1, row=6, padx=10, pady=10)
        
        # Print Portfolio Weights Button
        self.print_button = ttk.Button(self.frame, text='Print Portfolio Weights', command=self.print_values)
        self.print_button.grid(column=0, row=6, padx=10, pady=10)
        
        # Clear Results Button
        self.clear_button = ttk.Button(self.frame, text="Clear Results", command=self.clear_results)
        self.clear_button.grid(column=2, row=6, padx=10, pady=10)

    def render_correlation_matrix(self):
        """Renders the editable correlation matrix in the main frame"""
        self.matrix_popup = tk.Toplevel(self.root)
        self.matrix_popup.title("Edit Correlation Matrix")
        
        # Parse the current correlation matrix
        current_matrix = [list(map(float, row.split(','))) for row in self.correlation_matrix_var.get().split('\n')]

        self.entries = []
        for r in range(len(current_matrix)):
            row_entries = []
            for c in range(len(current_matrix)):
                e = ttk.Entry(self.matrix_popup, validate='key', font=('Inter', 12))
                e.insert(0, str(current_matrix[r][c]))
                e.grid(row=r, column=c, padx=5, pady=5)
                row_entries.append(e)
            self.entries.append(row_entries)

        # Update Button in Popup
        update_button = ttk.Button(self.matrix_popup, text='Update Correlation Matrix', command=self.update_correlation_matrix)
        update_button.grid(row=len(current_matrix), column=0, columnspan=len(current_matrix), padx=10, pady=10)
        
        # Bind the return key to trigger update button
        update_button.bind('<Return>', lambda event: self.update_correlation_matrix())

    def update_correlation_matrix(self):
        """Updates the correlation matrix with new values"""
        new_matrix = []
        for row_entries in self.entries:
            row = []
            for entry in row_entries:
                value = entry.get()
                row.append(value)
            new_matrix.append(', '.join(row))
        new_matrix_str = '\n'.join(new_matrix)
        self.correlation_matrix_var.set(new_matrix_str)
        self.matrix_popup.destroy()

    def optimize_portfolio(self):
        """Optimize the portfolio based on provided inputs and display the results."""
        try:
            portfolio_size = int(self.portfolio_size_entry.get())
            risk_aversion = float(self.risk_aversion_entry.get())
            risk_free_rate = float(self.risk_free_rate_entry.get())
            expected_returns = list(map(float, self.expected_returns_text.get("1.0", tk.END).strip().split(',')))
            volatilities = list(map(float, self.volatilities_text.get("1.0", tk.END).strip().split(',')))
            correlation_matrix = np.array([list(map(float, row.split(','))) for row in self.correlation_matrix_var.get().strip().split('\n')])

            if len(expected_returns) != portfolio_size or len(volatilities) != portfolio_size or correlation_matrix.shape != (portfolio_size, portfolio_size):
                raise ValueError("The number of expected returns, volatilities, or correlation matrix dimensions do not match the portfolio size.")
            
            covariance_matrix = np.outer(volatilities, volatilities) * correlation_matrix

            portfolio = OptimalPortfolio(portfolio_size, risk_aversion, risk_free_rate, expected_returns, covariance_matrix)
            frontier_weights, frontier_metrics = portfolio.calculate_mean_variance_efficient_frontier()
            
            self._values_for_print = (portfolio, frontier_weights, frontier_metrics)
            self.plot_efficient_frontier(frontier_metrics)
        except Exception as e:
            messagebox.showerror("Input Error", str(e))

    def plot_efficient_frontier(self, frontier_metrics):
        """Plot the efficient frontier based on calculated frontier metrics."""
        returns, risks, sharpe_ratios, utilities = zip(*frontier_metrics)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(risks, returns, 'b-', label='Efficient Frontier')
        ax.set_title('Mean-Variance Efficient Frontier')
        ax.set_xlabel('Risk (Standard Deviation)')
        ax.set_ylabel('Expected Return')
        ax.legend(loc='best')
        
        canvas = FigureCanvasTkAgg(fig, master=self.frame)
        canvas.draw()
        canvas.get_tk_widget().grid(column=0, row=9, columnspan=3, padx=10, pady=10, sticky=tk.W + tk.E)

    def clear_results(self):
        """Clear the plotted results and any optimization-related outputs."""
        for widget in self.frame.grid_slaves(row=9):
            widget.grid_forget()
        for widget in self.frame.grid_slaves(row=10):
            widget.grid_forget()
        self._values_for_print = None

    def print_values(self):
        """Print the calculated values"""
        if self._values_for_print:
            portfolio, frontier_weights, frontier_metrics = self._values_for_print
            min_var_weights = portfolio.calculate_minimum_variance_weights()
            print("Minimum Variance Portfolio Weights:", min_var_weights)
            for i, (weights, metrics) in enumerate(zip(frontier_weights, frontier_metrics)):
                print(f"Portfolio {i+1}:")
                print("Weights:", weights)
                print("Return, Risk, Sharpe Ratio, Utility:", metrics)
                print("-" * 30)
        else:
            messagebox.showinfo("Info", "No values to print. Perform an optimization first.")

def validate_correlation_entry(value):
    """Validates the correlation entry to ensure it is a float between -1 and 1"""
    try:
        float_val = float(value)
        return -1 <= float_val <= 1
    except ValueError:
        return False

if __name__ == "__main__":
    root = tk.Tk()
    app = PortfolioVisualising(root)
    root.mainloop()

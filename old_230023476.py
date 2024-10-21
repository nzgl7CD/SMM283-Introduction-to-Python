"""
@author: sebastianveum
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
        self.style.configure('TButton', font=('Helvetica', 12), padding=10, background='#4CAF50', foreground='#000000')
        self.style.map('TButton', background=[('active', '#45a049')])
        self._create_widgets_interface()
        self._values_for_print = None
        
    def _create_widgets_interface(self):
        """Create and arrange all the widgets in the application"""
        self.frame = ttk.Frame(self.root, padding="20")
        self.frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Portfolio Size
        ttk.Label(self.frame, text="Portfolio Size (2-12 securities):").grid(column=0, row=0, padx=10, pady=10, sticky=tk.W)
        self.portfolio_size_entry = ttk.Entry(self.frame, width=10, font=('Helvetica', 12))
        self.portfolio_size_entry.grid(column=1, row=0, padx=10, pady=10)
        self.portfolio_size_entry.insert(tk.END, "2")
        
        # Risk-Free Rate
        ttk.Label(self.frame, text="Risk-Free Rate:").grid(column=0, row=2, padx=10, pady=10, sticky=tk.W)
        self.risk_free_rate_entry = ttk.Entry(self.frame, width=10, font=('Helvetica', 12))
        self.risk_free_rate_entry.grid(column=1, row=2, padx=10, pady=10)
        self.risk_free_rate_entry.insert(tk.END, "0.045")

        # Risk Aversion
        ttk.Label(self.frame, text="Risk Aversion:").grid(column=0, row=1, padx=10, pady=10, sticky=tk.W)
        self.risk_aversion_entry = ttk.Entry(self.frame, width=10, font=('Helvetica', 12))
        self.risk_aversion_entry.grid(column=1, row=1, padx=10, pady=10)
        self.risk_aversion_entry.insert(tk.END, "3.0")

        # Expected Returns
        ttk.Label(self.frame, text="Expected Returns (comma-separated):").grid(column=0, row=3, padx=10, pady=10, sticky=tk.W)
        self.expected_returns_text = tk.Text(self.frame, width=60, height=1, font=('Helvetica', 12))
        self.expected_returns_text.grid(column=1, row=3, columnspan=2, padx=10, pady=10)
        self.expected_returns_text.insert(tk.END, "0.1934, 0.1575") # Suggest standards from excel file provided
        
        # Volatilities
        ttk.Label(self.frame, text="Volatilities (comma-separated):").grid(column=0, row=4, padx=10, pady=10, sticky=tk.W)
        self.volatilities_text = tk.Text(self.frame, width=60, height=1, font=('Helvetica', 12))
        self.volatilities_text.grid(column=1, row=4, columnspan=2, padx=10, pady=10)
        self.volatilities_text.insert(tk.END, "0.3025, 0.219") # Suggest standards from excel file provided

        #TODO improve design
        # Correlation Matrix
        ttk.Label(self.frame, text="Correlation Matrix (semicolon-separated rows):").grid(column=0, row=5, padx=10, pady=10, sticky=tk.W)
        self.correlation_matrix_text = tk.Text(self.frame, width=60, height=4, font=('Helvetica', 12))
        self.correlation_matrix_text.grid(column=1, row=5, columnspan=2, padx=10, pady=10)
        self.correlation_matrix_text.insert(tk.END, "1.0, 0.35; 0.35, 1.0") # Suggest standards from excel file provided

        # Button Frame to centre the buttons 
        button_frame = ttk.Frame(self.frame)
        button_frame.grid(column=0, row=6, columnspan=3, padx=10, pady=20, sticky=tk.W + tk.E)

        # Center buttons within the button_frame
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        button_frame.columnconfigure(2, weight=1)
    
    def update_fields(self, portfolio_size):
        """Update the volatility, expected returns, and correlation matrix fields"""
        self.volatilities_text.delete(1.0, tk.END)
        self.expected_returns_text.delete(1.0, tk.END)
        self.correlation_matrix_text.delete(1.0, tk.END)

        default_volatilities = ",".join(["0.1"] * portfolio_size)
        default_returns = ",".join(["0.1"] * portfolio_size)
        default_correlation = ";".join([",".join(["1.0" if i == j else "0.35" for j in range(portfolio_size)]) for i in range(portfolio_size)])
        
        self.volatilities_text.insert(tk.END, default_volatilities)
        self.expected_returns_text.insert(tk.END, default_returns)
        self.correlation_matrix_text.insert(tk.END, default_correlation)
    
    @property
    def values_for_print(self):
        return self._values_for_print
    
    @values_for_print.setter
    def values_for_print(self, values):
        self._values_for_print = values
        
    def plot_efficient_fronter(self, frontier_metrics):
        # Extracting frontier data from frontier_metrics
        frontier_risks = [metric[1] for metric in frontier_metrics]
        frontier_returns = [metric[0] for metric in frontier_metrics]
        sharpe_ratios = [metric[2] for metric in frontier_metrics]
        
        min_var_idx = np.argmin(frontier_risks)
        min_var_point = frontier_metrics[min_var_idx]
        
        max_sharpe_idx = np.argmax(sharpe_ratios)
        max_sharpe_point = frontier_metrics[max_sharpe_idx]
        
        # Plotting logic using matplotlib
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(frontier_risks, frontier_returns, 'b-o', label='Efficient Frontier')

        # Highlighting the min variance point
        ax.scatter(min_var_point[1], min_var_point[0], color='green', marker='o', s=100, 
                zorder=5, label=f'Min Variance Stdv: {min_var_point[1]:.4f}')

        # Highlighting the max Sharpe ratio point
        ax.scatter(max_sharpe_point[1], max_sharpe_point[0], color='red', marker='o', s=100, 
                zorder=5, label=f'Max Sharpe Ratio: {max_sharpe_point[2]:.4f}')
        
        # Set labels and title
        ax.set_title('Mean-Variance Efficient Frontier')
        ax.set_xlabel('Portfolio Risk')
        ax.set_ylabel('Portfolio Return')
        
        # Display legend
        ax.legend()
        
        # To show the portfolio metrics when the plot is closed
        def close_plot_and_show_metrics():
            self.plot_window.destroy()
            self.display_portfolio_key_values()
        
        self.plot_window = tk.Toplevel(self.root)
        self.plot_window.title('Efficient Frontier Plot')
        
        screen_width = self.plot_window.winfo_screenwidth()
        screen_height = self.plot_window.winfo_screenheight()
        window_width = 800
        window_height = 800
        
        x_position = int((screen_width - window_width) / 2)
        y_position = int((screen_height - window_height) / 2)
        
        # Set window size and position
        self.plot_window.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")
        
        # Embedding canvas in plot window
        canvas = FigureCanvasTkAgg(fig, master=self.plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)  # Expand canvas to fill window
        
        # Adding an Exit button to close the plot and show metrics
        ttk.Button(self.plot_window, text="Exit", command=close_plot_and_show_metrics).pack(side=tk.BOTTOM, pady=10)
        # Bind close event to destroy window and show metrics
        self.plot_window.protocol("WM_DELETE_WINDOW", close_plot_and_show_metrics)

    def display_portfolio_key_values(self):
        if self.values_for_print:
            top = tk.Toplevel(self.root)
            top.title("Portfolio Metrics")
        
            # Calculate and set the geometry of the dialog based on content size
            rows = len(self.values_for_print) + 1  # Including header row
            cols = 6 
            top.geometry(f"{cols * 135}x{rows * 150}")  
            
            formatted_str = "" 
            formatted_str += "{:<15s}{:^15s}{:^15s}{:^15s}{:^15s}{:^15s}\n".format("Portfolio Type","Portolfio No.", "Return", "Volatility", "Sharpe Ratio", "Utility")
            formatted_str += "-" * (92) + "\n"
            for key in self.values_for_print:
                formatted_str += "{:<15s}".format(key) 
                portfolio_index=self.values_for_print[key][0]
                return_value = self.values_for_print[key][1]
                volatility_value = self.values_for_print[key][2]
                sharpe_ratio_value = self.values_for_print[key][3]
                utility_value = self.values_for_print[key][4]
                formatted_str += "{:^15.0f}{:^15.4f}{:^15.4f}{:^15.4f}{:^15.4f}\n".format(portfolio_index+1, return_value, volatility_value, sharpe_ratio_value, utility_value)
            frame = ttk.Frame(top, padding=10)
            frame.pack(expand=True, fill='both')
            
            # Create a text widget with borders
            text_widget = tk.Text(frame, wrap=tk.NONE)
            text_widget.insert(tk.END, formatted_str)
            text_widget.configure(state='disabled', font=('Courier', 10), relief=tk.SOLID, borderwidth=1)
            text_widget.pack(expand=True, fill='both', padx=10, pady=10)

            # Button to close the window
            ttk.Button(frame, text="Close", command=top.destroy).pack(pady=10)
            
            # Print to terminal to satisfy the assignment requirement
            print(f'\n{formatted_str}')
    
    def exit_program(self):
        self.root.destroy()
        self.root.quit()


def validate_user_inputs(app, use_inputs: bool) -> bool:
    """Validate user inputs for volatilities, expected returns, and correlation matrix"""
    try:
        # Validate portfolio size
        portfolio_size = app.portfolio_size_entry.get()
        if not validate_input(portfolio_size, 2, 12, "Portfolio Size", int):
            return False

        # Validate risk aversion and risk-free rate
        risk_aversion = app.risk_aversion_entry.get()
        if risk_aversion and not validate_input(risk_aversion, 0.0, 20, "Risk Aversion", float):
            return False

        risk_free_rate = app.risk_free_rate_entry.get()
        if risk_free_rate and not validate_input(risk_free_rate, -1, 1, "Risk-Free Rate", float):
            return False

        if use_inputs:
            volatilities = app.volatilities_text.get("1.0", tk.END).strip().split(',')
            expected_returns = app.expected_returns_text.get("1.0", tk.END).strip().split(',')
            correlation_rows = app.correlation_matrix_text.get("1.0", tk.END).strip().split(';')

            if len(volatilities) != int(portfolio_size) or len(expected_returns) != int(portfolio_size) or len(correlation_rows) != int(portfolio_size):
                messagebox.showerror("Error", "The number of entries must match the portfolio size.")
                return False

            for vol in volatilities:
                if not validate_input(vol, 0.0, 1.0, "Volatility", float):
                    return False
            for ret in expected_returns:
                if not validate_input(ret, -1.0, 1.0, "Expected Return", float):
                    return False
            for row in correlation_rows:
                correlations = row.split(',')
                if len(correlations) != int(portfolio_size):
                    messagebox.showerror("Error", "The correlation matrix must be square and match the portfolio size.")
                    return False
                for corr in correlations:
                    if not validate_input(corr, -1.0, 1.0, "Correlation", float):
                        return False
        return True
    except ValueError:
        messagebox.showerror("Error", "Invalid input. Please check your entries.")
        return False

def validate_input(value, min_val, max_val, field_name, cast_type) -> bool:
    """Validate if the input value is within the specified range"""
    try:
        val = cast_type(value)
        if not (min_val <= val <= max_val):
            messagebox.showerror("Error", f"{field_name} must be between {min_val} and {max_val}.")
            return False
        return True
    except ValueError:
        messagebox.showerror("Error", f"Invalid input for {field_name}.")
        return False
    
def check_dataframe(dataframe)->bool:
    """Validates if dataframe is empty or only containes NaN"""
    if dataframe.empty or dataframe.isnull().all().all() or dataframe.isnull().any().any():
        messagebox.showerror("error", f"There's no optimal portfolio for this combination")
        return False
    else:
        return True

def click_update_fields(app):
    if validate_user_inputs(app, use_inputs=False):
        portfolio_size = int(app.portfolio_size_entry.get())
        app.update_fields(portfolio_size)

def click_optimiser(app, use_input: bool):
    if validate_user_inputs(app, use_input):
        portfolio_size = int(app.portfolio_size_entry.get())
        optimal_portfolio_object = generate_portfolio_object(app, use_input)
        frontier_weights, frontier_metrics = optimal_portfolio_object.calculate_mean_variance_efficient_frontier()
        dataframe = generate_dataframe(frontier_weights, frontier_metrics, portfolio_size)
        if check_dataframe(dataframe):
            write_to_excel(dataframe)
            app.values_for_print=generate_key_values_for_print(dataframe)
            
            app.plot_efficient_fronter(frontier_metrics)

def get_user_inputs(app, use_input: bool):
    """Gets all data used for the execution based on real data or regular optimising"""
    portfolio_size = int(app.portfolio_size_entry.get())
    risk_aversion = float(app.risk_aversion_entry.get()) if app.risk_aversion_entry.get() else 3.0
    risk_free_rate = float(app.risk_free_rate_entry.get()) if app.risk_free_rate_entry.get() else 0.045
    if use_input:
        volatilities = app.volatilities_text.get("1.0", tk.END).strip().split(',')
        expected_returns = app.expected_returns_text.get("1.0", tk.END).strip().split(',')
        correlation_rows = app.correlation_matrix_text.get("1.0", tk.END).strip().split(';')
        volatilities = list(map(float, volatilities))
        expected_returns = list(map(float, expected_returns))
        correlation_matrix = [list(map(float, row.split(','))) for row in correlation_rows]
        
        return portfolio_size, risk_aversion, risk_free_rate, volatilities, expected_returns, correlation_matrix
    
    return portfolio_size, risk_aversion, risk_free_rate
    
# For the display_portfolio_key_values() we could use the dataframe but wanted to show handling of dictionaries
def generate_key_values_for_print(dataframe):
    """Collects key portfolio metrics from the efficient frontier."""
    if dataframe is None:
        raise ValueError("Dataframe is not initialized. Please ensure the calculations are performed before calling this method.")
    df = dataframe
    
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
        'MinVar': [min_volatility_idx, min_volatility_return, min_volatility_volatility, min_volatility_sharpe, min_volatility_utility]
    }


def generate_dataframe(frontier_weights, frontier_metrics, portfolio_size):
    """Creates a dataframe with all values used to write in excel with correct decimal amount of 4 digits"""
    try:
        weight_columns = [f'w_{i+1}' for i in range(portfolio_size)]
        data = {
            'Return': [metric[0] for metric in frontier_metrics],
            'Volatility': [metric[1] for metric in frontier_metrics],
            'Sharpe Ratio': [metric[2] for metric in frontier_metrics],
            'Utility': [metric[3] for metric in frontier_metrics]
        }
        
        for i, col in enumerate(weight_columns):
            data[col] = [w[i] for w in frontier_weights]
        
        dataframe = pd.DataFrame(data)
        # Sort on return 
        dataframe.sort_values(by='Return', inplace=True)
        
        numeric_columns = ['Return', 'Volatility', 'Sharpe Ratio', 'Utility'] + weight_columns
        dataframe[numeric_columns] = dataframe[numeric_columns].round(4) # Round to 4 decimals
        return dataframe
    
    except KeyError as e:
        messagebox.showerror("error", f"Missing column in metrics: {e}")
        return False
    except ValueError as e:
        messagebox.showerror("error", str(e))
        return False
    except Exception as e:
        messagebox.showerror("error", f"An unexpected error occurred: {str(e)}")
        return False
        

def write_to_excel(dataframe):
    """Write the efficient frontier data to an Excel file."""
    output_file = '230023476PortfolioProblem.xlsx'
    # Write to excel sheet and replace existing output sheet if it exists
    with pd.ExcelWriter(output_file, mode='a', engine="openpyxl", if_sheet_exists="replace") as writer:
        dataframe.to_excel(writer, sheet_name='output', index=False)
        workbook = writer.book
        worksheet = workbook['output']
        
        # Fit cells with regular nested for loop as requested from assignment
        for column_cells in worksheet.columns:
            max_length = 0
            column = column_cells[0].column_letter  # Get the column name
            for cell in column_cells:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(cell.value)
                except:
                    pass
            adjusted_width = (max_length + 2) * 1.2  # Adjust the width for autofitting
            worksheet.column_dimensions[column].width = adjusted_width

def generate_portfolio_object(app, use_input: bool):
    """Generates a object to calculate the mean-variance optimal metrics with weights"""
    if use_input:
        portfolio_size, risk_aversion, risk_free_rate, volatilities, expected_returns, correlation_matrix = get_user_inputs(app, use_input)
        annual_covariance = calculate_covariance_matrix(volatilities, correlation_matrix)
    else:
        portfolio_size, risk_aversion, risk_free_rate = get_user_inputs(app, use_input)
        expected_returns, annual_covariance = get_real_world_data(portfolio_size)
    
    return OptimalPortfolio(portfolio_size, risk_aversion, risk_free_rate, expected_returns, annual_covariance)

def calculate_covariance_matrix(volatility, corr_matrix):
    corr_matrix = np.array(corr_matrix)
    stdv = np.array(volatility)
    return np.outer(stdv, stdv) * corr_matrix #Returns covariance matrix if user inputs are used

def get_real_world_data(portfolio_size):
    """Load and preprocess data from the Excel file."""
    ds = pd.read_excel('230023476PortfolioProblem.xlsx')
    monthly_return = ds.iloc[:, 1:portfolio_size + 1].pct_change()
    monthly_return.dropna()
    compounded_annual_returns = (monthly_return + 1).prod() ** (12 / len(monthly_return)) - 1
    annual_covariance = monthly_return.cov() * 12
    return compounded_annual_returns.values, annual_covariance

def main():
    root = tk.Tk()
    app = PortfolioVisualising(root)

    run_optimizer_button = ttk.Button(app.frame, text="Run Optimizer", command=lambda: click_optimiser(app, use_input=True), style='TButton')
    run_real_data_button = ttk.Button(app.frame, text="Run: Real Data", command=lambda: click_optimiser(app, use_input=False), style='TButton')
    exit_button = ttk.Button(app.frame, text="Exit", command=app.exit_program, style='TButton')
    update_fields_button = ttk.Button(app.frame, text="Update Fields", command=lambda: click_update_fields(app), style='TButton')

    # Grid buttons
    run_optimizer_button.grid(column=0, row=6, padx=5, pady=10)
    run_real_data_button.grid(column=1, row=6, padx=5, pady=10)
    exit_button.grid(column=2, row=6, padx=5, pady=10)
    update_fields_button.grid(column=2, row=0, padx=10, pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()
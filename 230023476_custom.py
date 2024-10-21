"""
@author: sebastianveum
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import customtkinter as ctk

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
        ctk.set_appearance_mode("dark")  
        ctk.set_default_color_theme("blue") 
        self._create_widgets_interface()
        self._values_for_print = None
        
    def _create_widgets_interface(self):
        """Create and arrange all the widgets in the application"""
        self.frame = ctk.CTkFrame(self.root, corner_radius=15)
        self.frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

        # Portfolio Size
        ctk.CTkLabel(self.frame, text="Portfolio Size (2-12 securities):").grid(column=0, row=0, padx=10, pady=10, sticky="w")
        self.portfolio_size_entry = ctk.CTkEntry(self.frame, width=120)
        self.portfolio_size_entry.grid(column=1, row=0, padx=10, pady=10)
        self.portfolio_size_entry.insert(tk.END, "2")
        
        # Risk-Free Rate
        ctk.CTkLabel(self.frame, text="Risk-Free Rate:").grid(column=0, row=2, padx=10, pady=10, sticky="w")
        self.risk_free_rate_entry = ctk.CTkEntry(self.frame, width=120)
        self.risk_free_rate_entry.grid(column=1, row=2, padx=10, pady=10)
        self.risk_free_rate_entry.insert(tk.END, "0.045")

        # Risk Aversion
        ctk.CTkLabel(self.frame, text="Risk Aversion:").grid(column=0, row=1, padx=10, pady=10, sticky="w")
        self.risk_aversion_entry = ctk.CTkEntry(self.frame, width=120)
        self.risk_aversion_entry.grid(column=1, row=1, padx=10, pady=10)
        self.risk_aversion_entry.insert(tk.END, "3.0")

        # Expected Returns
        ctk.CTkLabel(self.frame, text="Expected Returns (comma-separated):").grid(column=0, row=3, padx=10, pady=10, sticky="w")
        self.expected_returns_text = ctk.CTkTextbox(self.frame, width=400, height=30)
        self.expected_returns_text.grid(column=1, row=3, columnspan=2, padx=10, pady=10)
        self.expected_returns_text.insert(tk.END, "0.1934, 0.1575") # Suggest standards from excel file provided
        
        # Volatilities
        ctk.CTkLabel(self.frame, text="Volatilities (comma-separated):").grid(column=0, row=4, padx=10, pady=10, sticky="w")
        self.volatilities_text = ctk.CTkTextbox(self.frame, width=400, height=30)
        self.volatilities_text.grid(column=1, row=4, columnspan=2, padx=10, pady=10)
        self.volatilities_text.insert(tk.END, "0.3025, 0.219") # Suggest standards from excel file provided

        # Correlation Matrix (adjusted as suggested by Adrian)
        self.correlation_matrix_button = ctk.CTkButton(self.frame, text='Edit Correlation Matrix', command=self.open_correlation_popup)
        self.correlation_matrix_button.grid(column=1, row=5, padx=10, pady=10)

        ctk.CTkLabel(self.frame, text="Correlation Matrix:").grid(column=0, row=5, padx=10, pady=10, sticky="w")
        self.correlation_matrix_var = tk.StringVar()
        self.correlation_matrix_var.set("1, 0.35\n0.35, 1")
        
        self.correlation_matrix_label = ctk.CTkLabel(self.frame, textvariable=self.correlation_matrix_var, wraplength=400, justify=tk.LEFT)
        self.correlation_matrix_label.grid(column=2, row=5, columnspan=2, padx=10, pady=10, sticky="w")

        # Button Frame to centre the buttons 
        button_frame = ctk.CTkFrame(self.frame)
        button_frame.grid(column=0, row=6, columnspan=3, padx=10, pady=20, sticky="we")

        # Center buttons within the button_frame
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        button_frame.columnconfigure(2, weight=1)
    
    def open_correlation_popup(self):
        # Create popup window
        popup = ctk.CTkToplevel(self.root)
        popup.title("Edit Correlation Matrix")
        
        # Make popup transient
        popup.transient(self.root)
        popup.grab_set()
        
        # Parse the current correlation matrix
        current_matrix_str = self.correlation_matrix_var.get()
        self.current_matrix = [list(map(float, row.split(','))) for row in current_matrix_str.split('\n')]
        rows = len(self.current_matrix)
        cols = len(self.current_matrix[0])
        
        self.entries = []
        float_validator = (popup.register(validate_correlation_entry), '%P')
        
        for r in range(rows):
            row_entries = []
            for c in range(cols):
                if r == c:
                    lbl = ctk.CTkLabel(popup, text=str(self.current_matrix[r][c]), width=15, justify=tk.CENTER, fg_color="#D3D3D3")
                    lbl.grid(row=r, column=c, padx=5, pady=5)
                    row_entries.append(None)  # No Entry widget here
                else:
                    e = ctk.CTkEntry(popup, width=40, validate='key', validatecommand=float_validator, justify=tk.CENTER)
                    e.insert('end', self.current_matrix[r][c])
                    e.grid(row=r, column=c, padx=5, pady=5)
                    row_entries.append(e)
            self.entries.append(row_entries)

        save_button = ctk.CTkButton(popup, text='Save', command=lambda: self.save_correlation_matrix(popup))
        save_button.grid(row=rows+1, column=0, columnspan=cols, pady=10)

        # Adjust position
        popup.update_idletasks()
        width = popup.winfo_width()
        height = popup.winfo_height()
        x_offset = (popup.winfo_screenwidth() - width) // 2
        y_offset = (popup.winfo_screenheight() - height) // 3
        popup.geometry(f"+{x_offset}+{y_offset}")
    
    def save_correlation_matrix(self, popup):
        """Save correlation matrix to the label on main page"""
        matrix = []
        for r, row in enumerate(self.entries):
            matrix_row = []
            for c, e in enumerate(row):
                if e is None:  # This is a Label widget
                    matrix_row.append(str(self.current_matrix[r][c]))
                else:
                    matrix_row.append(e.get() if e.get() != "" else "0.35")
            matrix.append(matrix_row)
        
        # Validate the matrix entries
        for r in range(len(matrix)):
            for c in range(len(matrix[r])):
                if not validate_correlation_entry(matrix[r][c]):
                    return
        
        # Convert to comma-separated string and round numbers
        matrix_str = '\n'.join([', '.join([f"{float(value):.0f}" if float(value) == 1 else f"{float(value):.2f}" for value in row]) for row in matrix])
        
        # Update the main window label
        self.correlation_matrix_var.set(matrix_str)

        popup.destroy()
    
    def update_fields(self, portfolio_size):
        """Update the volatility, expected returns, and correlation matrix fields"""
        self.volatilities_text.delete(1.0, tk.END)
        self.expected_returns_text.delete(1.0, tk.END)
        
        # Clear previous correlation matrix
        self.correlation_matrix_var.set("")
        
        # Default values
        default_volatilities = ",".join(["0.1"] * portfolio_size)
        default_returns = ",".join(["0.1"] * portfolio_size)
        default_correlation = ";".join([",".join(["1.0" if i == j else "0.35" for j in range(portfolio_size)]) for i in range(portfolio_size)])
        
        # Update text fields
        self.volatilities_text.insert(tk.END, default_volatilities)
        self.expected_returns_text.insert(tk.END, default_returns)
        
        # Update correlation matrix
        self.correlation_matrix_var.set(default_correlation.replace(";", "\n"))  
        self.correlation_matrix_label.configure(text=self.correlation_matrix_var.get())  # Update label content
    
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
                zorder=5, label=f'Min Variance Stdv: {min_var_point[1]:.3f}')

        # Highlighting the max Sharpe ratio point
        ax.scatter(max_sharpe_point[1], max_sharpe_point[0], color='red', marker='o', s=100, 
                zorder=5, label=f'Max Sharpe Ratio: {max_sharpe_point[2]:.3f}')
        
        # Set labels and title
        ax.set_title('Mean-Variance Efficient Frontier')
        ax.set_xlabel('Portfolio Risk')
        ax.set_ylabel('Portfolio Return')
        ax.legend()
        
        # To show the portfolio metrics when the plot is closed
        def close_plot_and_show_metrics():
            self.plot_window.destroy()
            self.display_portfolio_key_values()
        
        self.plot_window = ctk.CTkToplevel(self.root)
        self.plot_window.title('Efficient Frontier Plot')
        
        # Make plot window transient and set it above the main window
        self.plot_window.transient(self.root)
        self.plot_window.attributes('-topmost', True)  # Set plot window to be on top
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
        ctk.CTkButton(self.plot_window, text="Exit", command=close_plot_and_show_metrics).pack(side=tk.BOTTOM, pady=10)
        
        # Bind close event to destroy window and show metrics
        self.plot_window.protocol("WM_DELETE_WINDOW", close_plot_and_show_metrics)

    def display_portfolio_key_values(self):
        if self.values_for_print:
            top = ctk.CTkToplevel(self.root)
            top.title("Portfolio Metrics")
        
            # Calculate and set the geometry of the dialog based on content size
            rows = len(self.values_for_print) + 1  # Including header row
            cols = 6 
            top.geometry(f"{cols * 135}x{rows * 150}")  
            
            formatted_str = "" 
            formatted_str += "{:<15s}{:^15s}{:^15s}{:^15s}{:^15s}{:^15s}\n".format("Portfolio Type","Portolfio No.", "Return", "Volatility", "Sharpe Ratio", "Utility")
            formatted_str += "-" * (95) + "\n"
            for key in self.values_for_print:
                formatted_str += "{:<15s}".format(key) 
                portfolio_index=self.values_for_print[key][0]
                return_value = self.values_for_print[key][1]
                volatility_value = self.values_for_print[key][2]
                sharpe_ratio_value = self.values_for_print[key][3]
                utility_value = self.values_for_print[key][4]
                formatted_str += "{:^15.0f}{:^15.4f}{:^15.4f}{:^15.4f}{:^15.4f}\n".format(portfolio_index+1, return_value, volatility_value, sharpe_ratio_value, utility_value)
            frame = ctk.CTkFrame(top, corner_radius=15)
            frame.pack(expand=True, fill='both', padx=10, pady=10)
            
            # Create a text widget with borders
            text_widget = tk.Text(frame, wrap=tk.NONE)
            text_widget.insert(tk.END, formatted_str)
            text_widget.configure(state='disabled', font=('Courier', 12), relief=tk.SOLID, borderwidth=1)
            text_widget.pack(expand=True, fill='both', padx=10, pady=10)

            # Button to close the window
            ctk.CTkButton(frame, text="Close", command=top.destroy).pack(pady=10)
            
            # Print to terminal to satisfy the assignment requirement
            print(f'\n{formatted_str}')


def validate_correlation_entry(value):
        try:
            if value.strip() == "" or value.strip()== "-":
                return True  # Allow empty strings
            float_value = float(value)
            return -1.0 <= float_value <= 1.0
        except ValueError:
            return False 
        
def validate_user_inputs(app, use_inputs: bool) -> bool:
    """Validate user inputs for volatilities, expected returns, and correlation matrix"""
    try:
        # Validate portfolio size
        portfolio_size = int(app.portfolio_size_entry.get())
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
            correlation_matrix_str = app.correlation_matrix_var.get().strip()
            correlation_rows = correlation_matrix_str.split('\n')
            
            if len(volatilities) != int(portfolio_size) or len(expected_returns) != int(portfolio_size) or len(correlation_rows) != int(portfolio_size):
                messagebox.showerror("Error", "The number of entries must match the portfolio size.")
                return False

            for vol in volatilities:
                if not validate_input(vol, 0.0, 1.0, "Volatility", float):
                    return False
            for ret in expected_returns:
                if not validate_input(ret, 0, 1.0, "Expected Return", float):
                    return False
            for row in correlation_rows:
                    correlations = row.split(',')
                    if len(correlations) != portfolio_size:
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
        correlation_matrix_str = app.correlation_matrix_var.get().strip()
        correlation_rows = correlation_matrix_str.split('\n')
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
    root = ctk.CTk()
    app = PortfolioVisualising(root)

    # Grid buttons
    run_optimizer_button = ctk.CTkButton(app.frame, text="Run Optimizer", command=lambda: click_optimiser(app,use_input=True))
    run_real_data_button = ctk.CTkButton(app.frame, text="Run: Real Data", command=lambda: click_optimiser(app,use_input=False))
    exit_button = ctk.CTkButton(app.frame, text="Exit", command=lambda:exit())
    update_fields_button = ctk.CTkButton(app.frame, text="Update Fields", command=lambda:click_update_fields(app))

    # Configure button style
    button_font = ('Inter', 12)
    button_style = 'TButton'  # Assuming 'TButton' is defined in customtkinter

    run_optimizer_button.configure(font=button_font)
    run_real_data_button.configure(font=button_font)
    exit_button.configure(font=button_font)
    update_fields_button.configure(font=button_font)

    # Grid buttons
    run_optimizer_button.grid(column=0, row=6, padx=5, pady=10)
    run_real_data_button.grid(column=1, row=6, padx=5, pady=10)
    exit_button.grid(column=2, row=6, padx=5, pady=10)
    update_fields_button.grid(column=2, row=0, padx=10, pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()

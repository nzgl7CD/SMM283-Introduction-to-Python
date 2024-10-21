#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import ttk, messagebox
from openpyxl import load_workbook
from openpyxl.drawing.image import Image

def run_analysis():
    try:
        num_assets = int(asset_spinbox.get())
        if not 2 <= num_assets <= 12:
            raise ValueError("Number of assets must be between 2 and 12.")
        
        # Retrieve user inputs
        ex_returns_values = [float(val) for val in ex_returns_entry.get().split(',')]
        vol_values = [float(val) for val in vol_entry.get().split(',')]
        corr_values = [float(val) for val in corr_entry.get().split(',')]
        risk_adv = float(risk_adv_entry.get())
        risk_free = float(risk_free_entry.get())
        
        if len(ex_returns_values) != num_assets or len(vol_values) != num_assets or len(corr_values) != num_assets**2:
            raise ValueError("Incorrect number of inputs for returns, volatilities, or correlation coefficients.")

        # Load the CSV file
        df = pd.read_csv('ian\Python CW.csv', header=0)
        df = df.dropna(axis=1)
        df = df.drop(columns=["Date"])

        tickers = {
            1: "Nvidia",
            2: "AAPL",
            3: "BAESY",
            4: "CJJD",
            5: "AZN",
            6: "SAS",
            7: "TSLA",
            8: "Saab AB",
            9: "PFE",
            10: "GME",
            11: "HSBC",
            12: "LVMH"
        }

        desired_columns = list(range(1, num_assets + 1))
        desired_tickers = [tickers[i] for i in desired_columns]
        df = df[desired_tickers]

        ex_returns = pd.Series(ex_returns_values, index=desired_tickers)
        vol = pd.Series(vol_values, index=desired_tickers)
        corr_matrix = np.array(corr_values).reshape((num_assets, num_assets))
        corr = pd.DataFrame(corr_matrix, index=desired_tickers, columns=desired_tickers)
        
        cov = corr.mul(vol, axis=0).mul(vol, axis=1)
        inv_cov = np.linalg.inv(cov)
        l_matrix = ex_returns @ inv_cov

        user_vector = 1
        vector = pd.Series([user_vector] * len(desired_columns))
        m_matrix = vector @ inv_cov

        a = vector @ l_matrix
        b = ex_returns @ l_matrix
        c = vector @ m_matrix
        d = b * c - (a ** 2)

        g = 1/(d) * ((m_matrix*b) - (l_matrix*a))
        h = 1/(d) * ((l_matrix*c) - (m_matrix*a))
        gh = g + h

        # optimum_weights = g + (target_return * h)

        # print("l_matrix:", l_matrix)
        # print("m_matrix:", m_matrix)
        # print("a:", a)
        # print("b:", b)
        # print("c:", c)
        # print("d:", d)
        # print("g:", g)
        # print("h:", h)
        # print("gh:", gh)
        # print("optimum_weights:", optimum_weights)

        # return_optimum_point = optimum_weights @ ex_returns
        # risk_optimum_point = (optimum_weights.T @ cov @ optimum_weights) ** 0.5

        # print("return_optimum_point:", return_optimum_point)
        # print("risk_optimum_point:", risk_optimum_point)

        return_g = g @ ex_returns
        risk_g = (g.T @ cov @ g) ** 0.5

        return_h = h @ ex_returns
        risk_h = (h.T @ cov @ h) ** 0.5

        return_gh = gh @ ex_returns
        risk_gh = (gh.T @ cov @ gh) ** 0.5

        print(f"risk_g={risk_g}\nrisk_h={risk_h}\nrisk_g+h={risk_gh}")

        min_var = a/c
        opt_var = min_var - (d/(c**2))/(risk_free - min_var)

        print("min_var:", min_var)
        print("opt_var:", opt_var)

        portfolio_returns = []
        portfolio_risk = []
        portfolio_weights = []
        portfolio_utility = []
        portfolio_sharpe_ratios = []

        p_returns = np.linspace(0, 1, 101)

        for i in p_returns:
            p_weights = g + (i * h)
            p_return = np.sum(p_weights * ex_returns)
            portfolio_variance = np.dot(p_weights.T, np.dot(cov, p_weights))
            p_risk = (p_weights @ cov @ p_weights.T) ** 0.5
            excess_return = p_return - risk_free
            p_sharpe_ratios = excess_return / p_risk
            p_utility = p_return - (0.5 * risk_adv * portfolio_variance)

            portfolio_weights.append(p_weights)
            portfolio_risk.append(p_risk)
            portfolio_utility.append(p_utility)
            portfolio_sharpe_ratios.append(p_sharpe_ratios)
            portfolio_returns.append(p_return)

        portfolio_returns = np.array(portfolio_returns)
        portfolio_risk = np.array(portfolio_risk)
        portfolio_weights = np.array(portfolio_weights)
        portfolio_utility = np.array(portfolio_utility)

        max_sharpe = max(portfolio_sharpe_ratios)

        portfolio_capital_allocations = []

        for i in p_returns:
            p_weights = g + (i * h)
            p_risk = (p_weights @ cov @ p_weights.T) ** 0.5
            p_capital_allocation = risk_free + (max_sharpe * p_risk)
            portfolio_capital_allocations.append(p_capital_allocation)

        portfolio_capital_allocations = np.array(portfolio_capital_allocations)

        weight_columns = [f'{desired_tickers[i]} Weight' for i in range(len(desired_columns))]
        data = {
            'Return': [p_ret for p_ret in portfolio_returns],
            'Volatility': [risk for risk in portfolio_risk],
            'Sharpe Ratio': [sharpe for sharpe in portfolio_sharpe_ratios],
            'Utility': [utility for utility in portfolio_utility],
            'Capital Allocation Line Ret': [capital for capital in portfolio_capital_allocations]
        }

        for i, col in enumerate(weight_columns):
            data[col] = [w[i] for w in portfolio_weights]

        df = pd.DataFrame(data)

        df["Capital Allocation Line Ret"] = risk_free + (max_sharpe * df["Volatility"])

        # Define the path to save the Excel file on the desktop
        #excel_file = 'portfolio_analysis.xlsx'
        #df.to_excel(excel_file, index=False)
        #messagebox.showinfo("Success", f"Data saved to {excel_file}")

        plot_efficient_frontier(portfolio_risk,portfolio_returns, portfolio_sharpe_ratios)

    except Exception as e:
        messagebox.showerror("Error", str(e))

def plot_efficient_frontier(frontier_risks, frontier_returns, sharpe_ratios):
    min_var_idx = np.argmin(frontier_risks)
    max_sharpe_idx = np.argmax(sharpe_ratios)

    plt.figure(figsize=(12, 8))
    plt.plot(frontier_risks, frontier_returns, 'b-o', label='Efficient Frontier')

    plt.scatter(frontier_risks[min_var_idx], frontier_returns[min_var_idx], color='green', marker='o', s=100, 
    zorder=5, label=f'Min Variance Portfolio: {frontier_risks[min_var_idx]:.2f}')

    plt.scatter(frontier_risks[max_sharpe_idx], frontier_returns[max_sharpe_idx], color='red', marker='o', s=100, 
        zorder=5, label=f'Optimal Portfolio: {sharpe_ratios[max_sharpe_idx]:.2f}')

    plt.xlabel('Portfolio Volatility', fontsize=12, fontweight='bold')
    plt.ylabel('Portfolio Return', fontsize=12, fontweight='bold')
    plt.title('Mean-Variance Efficient Frontier', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')

    plt.tight_layout()  # Ensures labels do not overlap
    plot_filename = 'plot.png'
    plt.savefig(plot_filename)
    plt.show()

  

def output_to_excel_file(df, plot_filename):
   
    sorted_df = df.round(4)

    sorted_df_Utility = sorted_df.sort_values(by='Utility', ascending =False)
    max_utility= sorted_df_Utility.iloc[1]
    print("maximum utility portfolio")

    sorted_df_sharpe = sorted_df.sort_values(by='Sharpe Ratio', ascending =False)
    max_sharpe= sorted_df_sharpe.iloc[1]
    print("maximum utility portfolio")


    excel_filename = 'output.xlsx'
    sorted_df.to_excel(excel_filename, index=False, sheet_name='Output DataFrame', engine='openpyxl')
    
    workbook = load_workbook(excel_filename)
    worksheet = workbook['Output DataFrame']

    img = Image (plot_filename)
    worksheet.add_image(img, 'H2')

    workbook.save(excel_filename)

    print(f'DataFrame and plot have been successfully dave to {excel_filename}')


# Create the main window
root = Tk()
root.title("Asset Selection")

# Add a label and a spinbox for the number of assets
label = Label(root, text="Select number of assets (2-12):")
label.pack(pady=10)

asset_spinbox = ttk.Spinbox(root, from_=2, to=12, increment=1)
asset_spinbox.pack(pady=10)

# Add entries for expected returns, volatility, correlation, risk aversion, and risk-free rate
ex_returns_label = Label(root, text="Expected Rates of Return (comma separated):")
ex_returns_label.pack(pady=5)
ex_returns_entry = Entry(root, width=50)
ex_returns_entry.pack(pady=5)

vol_label = Label(root, text="Volatility (comma separated):")
vol_label.pack(pady=5)
vol_entry = Entry(root, width=50)
vol_entry.pack(pady=5)

corr_label = Label(root, text="Correlation Coefficients (comma separated, row-wise):")
corr_label.pack(pady=5)
corr_entry = Entry(root, width=50)
corr_entry.pack(pady=5)

risk_adv_label = Label(root, text="Risk Aversion Coefficient:")
risk_adv_label.pack(pady=5)
risk_adv_entry = Entry(root, width=50)
risk_adv_entry.pack(pady=5)

risk_free_label = Label(root, text="Risk-Free Rate of Return:")
risk_free_label.pack(pady=5)
risk_free_entry = Entry(root, width=50)
risk_free_entry.pack(pady=5)

# Add a button to run the analysis
button = Button(root, text="Run Analysis", command=run_analysis)
button.pack(pady=20)

# Run the Tkinter event loop
root.mainloop()





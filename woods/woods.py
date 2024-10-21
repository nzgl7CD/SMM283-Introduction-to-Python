import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


returns=[0.1924,0.1575]
vol=[0.3025,0.219]
corr=[[1,0.35],[0.35,1]]
risk_adv=3
risk_free=0.045
n=2

cov=np.outer(vol, vol) * corr

cov=np.array(cov)
vol=np.array(vol)
returns=np.array(returns)

inv_cov=np.linalg.inv(cov)
u = np.ones(n)
l=returns @ inv_cov
m=u @ inv_cov
a = sum([sum(u[i] * returns[j] * inv_cov[i, j] for i in range(n)) for j in range(n)])
b = sum([sum(returns[i] * returns[j] * inv_cov[i, j] for i in range(n)) for j in range(n)])
c = sum([sum(u[i] * u[j] * inv_cov[i, j] for i in range(n)) for j in range(n)])
d = b*c-a**2

g = (m*b-l*a)/d
h = (l*c-m*a)/d

# print(A)

portfolio_returns = []
portfolio_risk = []
portfolio_weights = []
portfolio_utility = []
portfolio_sharpe_ratio = []

p_returns = np.linspace(0, 1, 101)

for i in p_returns:
    p_weights=g+(i*h)
    p_return = np.sum(p_weights * returns)
    portfolio_variance = np.dot(p_weights.T, np.dot(cov, p_weights))
    p_risk = np.sqrt(portfolio_variance)
    excess_return = p_return - risk_free
    p_sharpe_ratio = excess_return / portfolio_risk
    p_utility = p_return - (0.5 * risk_adv * portfolio_variance)
    
    portfolio_weights.append(p_weights)
    portfolio_risk.append (p_risk)
    portfolio_utility.append (p_utility)
    portfolio_sharpe_ratio.append (p_sharpe_ratio)
    portfolio_returns.append (p_returns)
    
    

portfolio_returns = np.array(portfolio_returns)
portfolio_risk = np.array(portfolio_risk)
portfolio_weights = np.array(portfolio_weights)
# portfolio_utility = np.array(portfolio_utility)
# portfolio_sharpe_ratio=np.array(portfolio_sharpe_ratio)

weight_columns = [f'w_{i+1}' for i in range(n)]
data = {
            'Return': [p_ret for p_ret in portfolio_returns],
            'Volatility': [risk for risk in portfolio_risk],
            'Sharpe Ratio': [sharpe for sharpe in portfolio_sharpe_ratio],
            'Utility': [utilitiy for utilitiy in portfolio_utility]
        }


for i, col in enumerate(weight_columns):
    data[col] = [w[i] for w in portfolio_weights]

df = pd.DataFrame(data)
# excel_file = 'portfolio_analysis.xlsx'
# df.to_excel(excel_file, index=False)
# print(f"Data saved to {excel_file}")

# plt.figure(figsize=(10, 6))
# plt.line(portfolio_risks, portfolio_returns, c=sharpe_ratios, cmap='viridis')
# plt.colorbar(label='Sharpe Ratio')
# plt.title('Efficient Frontier')
# plt.xlabel('Risk (Standard Deviation)')
# plt.ylabel('Return')
# plt.show()



    

    

    

portfolio_returns = np.array(portfolio_returns)

portfolio_risk = np.array(portfolio_risk)

portfolio_weights = np.array(portfolio_weights)


 

print (portfolio_returns)

print (portfolio_risk)

print (portfolio_weights)

print (portfolio_utility)

print (portfolio_sharpe_ratio)



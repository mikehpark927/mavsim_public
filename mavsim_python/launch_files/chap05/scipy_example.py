from scipy.optimize import minimize
import numpy as np

def objective(x):
    return(x[0]**2 + x[1]**2)

x0 = np.array([5,2]) # initial condition
cons = ({'type': 'eq', 
         'fun': lambda x: np.array([x[0]+x[1]-2]),
        })

res = minimize(objective, x0, method='SLSQP', constraints=cons)
print("xstar =", res.x)
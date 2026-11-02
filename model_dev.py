import pandas as pd
import numpy as np




# load historical customer data
customer_data = pd.read_csv('./data/customer_data.csv', index_col=None)
customer_claims = pd.read_csv('./data/customer_claims.csv', index_col=None)

# write your code to develop your agent program here.
# you can, of course, add additional .py files as needed.

# randomly assign instances to training and test sets (50% each for illustration purposes - 66.6% training maybe more typical)
train_idx = np.random.random(size=customer_data.shape[0]) < 0.5

x_train = customer_data.loc[train_idx]
y_train = customer_claims.loc[train_idx]
x_test = customer_data.loc[~train_idx]
y_test = customer_claims.loc[~train_idx]
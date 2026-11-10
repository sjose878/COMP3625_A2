import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from joblib import load

BUDGET = 1000000
PREMIUM_MULTIPLIER = 1.2 #20% on top of coverage

# load historical customer data
customer_data = pd.read_csv('./data/customer_data.csv', index_col=None)
customer_claims = pd.read_csv('./data/customer_claims.csv', index_col=None)

# write your code to develop your agent program here.
# you can, of course, add additional .py files as needed.
"""
# randomly assign instances to training and test sets (50% each for illustration purposes - 66.6% training maybe more typical)
train_idx = np.random.random(size=customer_data.shape[0]) < 0.5

x_train = customer_data.loc[train_idx]
y_train = customer_claims['claim'].loc[train_idx]
x_test = customer_data.loc[~train_idx]
y_test = customer_claims['claim'].loc[~train_idx]
#model = LogisticRegression(max_iter=1000).fit(X=x_train,y=y_train)
#model = KNeighborsClassifier(n_neighbors=1).fit(X=x_train, y=y_train)

#temp ^ need trained model
"""
model = load("trained_model.joblib")
reduced_features = load("reduced_features.joblib")

# At grading time, I will call the below function with a dataframe of new potential customers
# The dataframe I pass in will have the same columns as customer_data.csv, but will not include any of the customers in customer_data.csv
# You will not be given any of the historical customer data or claims data at grading time, 
# so you will need to save any models you train in model_dev.py and load them here.
def utility_curve(dollars: int):
    """
    Takes the given dollar amount and returns the current customer's utility for that amount. 
    :param dollars: an integer representing the given dollar amount.
    :return: a float representing the customers utility for the dollar amount. 
    """
    if dollars + 150000 <= 0:
        return -263.31 + 22.09 * np.log(-149999 + 150000)
 
    return -263.31 + 22.09 * np.log(dollars + 150000)

def set_premium(customer_data: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts a dataframe of data on potential customers, identifies which customers to take on, and sets premiums for them.
    This function may load a saved model (previously trained using supervised learning) to save time.
    :param customer_data: a dataframe with the same columns as customer_data.csv
    :return: a DataFrame of the same length as customer_data, with the following columns:
            p_claim:  your estimated probability that each customer in customer_data will make a claim
            desired_customer:  a boolean column indicating which customers you want to take on
            premium:    the amount of the premium to charge each customer
    """
    # https://stackoverflow.com/questions/10712002/create-an-empty-list-with-certain-size-in-python
    customer_premiums = [None] * len(customer_data[reduced_features])
    total_coverage_amount = 0

    probabilities = model.predict_proba(customer_data[reduced_features])

    sorted_customers = []
    total_profit = 0

    # calculated needed values for customers and stores them
    for i in range(len(customer_data[reduced_features])):
        coverage = int(customer_data.loc[i, "coverage_amount"])
        p_no_claim = probabilities[i][0]
        p_claim = probabilities[i][1]
        premium = PREMIUM_MULTIPLIER * (p_claim * coverage) #(p_claim * coverage) from insurer's eu
        expected_profit = premium - (p_claim * coverage)

        sorted_customers.append({"index": i,
                                "coverage": coverage,
                                "p_no_claim": p_no_claim,
                                "p_claim": p_claim,
                                "premium": premium,
                                "expected_profit": expected_profit,
                                })

    # sorts customers by highest expected profit to lowest expected profit
    sorted_customers.sort(key=lambda x: x["expected_profit"], reverse=True)

    #calculates uninsured and insured utility for each customer then decides if the customer is a desired customer 
    for i in range(len(sorted_customers)):
        desired_customer = False

        eu_uninsured = sorted_customers[i]["p_no_claim"] * utility_curve(0) + sorted_customers[i]["p_claim"] * utility_curve(-(sorted_customers[i]["coverage"])) #probability of no claim -> wont lose anything so 0. probability of claim -> will lose full coverage
        eu_insured = sorted_customers[i]["p_no_claim"] * utility_curve(-(sorted_customers[i]["premium"])) + sorted_customers[i]["p_claim"] * utility_curve(-(sorted_customers[i]["premium"])) #probability of no claim -> wont lose anything so 0 lost but still pay premium. probability of claim -> insurance reimburses coverage amount but still have to pay premium

        if (total_coverage_amount + sorted_customers[i]["coverage"] < BUDGET) and (eu_insured > eu_uninsured) and (sorted_customers[i]["expected_profit"] > 0):
            desired_customer = True
            total_coverage_amount += sorted_customers[i]["coverage"]
            total_profit += sorted_customers[i]["expected_profit"]
  
        customer_premiums[sorted_customers[i]["index"]] = [sorted_customers[i]["p_claim"], desired_customer, sorted_customers[i]["premium"]]

    print(f'total budget used: {total_coverage_amount}')
    print(f'total profit: {total_profit}')
    return pd.DataFrame(customer_premiums, columns=["p_claim", "desired_customer", "premium"])

result = set_premium(customer_data)
print(result[result["desired_customer"] == True])

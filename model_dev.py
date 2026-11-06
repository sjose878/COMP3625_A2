import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier



    
# load historical customer data
customer_data = pd.read_csv('./data/customer_data.csv', index_col=None)
customer_claims = pd.read_csv('./data/customer_claims.csv', index_col=None)

# write your code to develop your agent program here.
# you can, of course, add additional .py files as needed.

"""
Takes the wine dataset and recursively discards features one at a time to improve accuracy of classification for LogisticRegression()

param: features_list - list of features from wine dataset. Since it's recursive, the parameter may also be the list of features that are redudant for classification
param: best_score - accuracy score that needs to be surpassed to determine if another feature should be recursively removed. 0 is passed in initially
"""
def feature_selection(features_list, best_score):
    best_model = None
    accuracy_improved = False
    # try every combo of every feature except one
    for i in range(len(features_list)): 
        # create a model with one less feature
        x_minus = x_train[features_list].drop(features_list[i], axis=1)
        x_test_minus = x_test[features_list].drop(features_list[i], axis=1)
        new_model = LogisticRegression(solver='newton-cholesky').fit(X=x_minus, y=y_train)
        # see how well the new model does
        score = new_model.score(x_test_minus, y_test)

        if score > best_score:
            best_score = score
            best_model = new_model
            feature_dropped = features_list[i]
            print(f'Feature dropped: {feature_dropped}')
            accuracy_improved = True

    if accuracy_improved:
        # drop insignificant feature.
        features_list = features_list.drop(feature_dropped)
        # Recursively remove another feature (if improvement is found).
        best_model, best_score = feature_selection(features_list, best_score)

        if best_model is None:
            return new_model, best_score
        
        return best_model, best_score
    else:
        # Return model & score if no improvement is found from removing a feature.
        print("No more improvement can be found!")
        return best_model, best_score

# MAIN CODE

# randomly assign instances to training and test sets. (66.6-33.3 split)
train_idx = np.random.random(size=customer_data.shape[0]) < 0.9

x_train = customer_data.loc[train_idx]
y_train = customer_claims['claim'].loc[train_idx]
x_test = customer_data.loc[~train_idx]
y_test = customer_claims['claim'].loc[~train_idx]
print(y_train)

neighbor_model = KNeighborsClassifier(n_neighbors=27).fit(X=x_train, y=y_train)

lr_model = LogisticRegression(solver='newton-cholesky').fit(X=x_train, y=y_train)
print(f'Accuracy with hello neighbor: {neighbor_model.score(x_test, y_test)}')
print(f'Accuracy with logistic regression model: {lr_model.score(x_test,y_test)}')


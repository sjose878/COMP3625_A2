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
Code is repurposed from lab 6
param: features_list - list of features from wine dataset. Since it's recursive, the parameter may also be the list of features that are redudant for classification
param: best_score - accuracy score that needs to be surpassed to determine if another feature should be recursively removed. 0 is passed in initially
"""
def feature_selection(features_list, best_score: float):
    best_model = None
    feature_dropped = None
    accuracy_improved = False
    best_curr_score = best_score

    # try every combo of every feature except one
    for i in range(len(features_list)): 

        # create a model with one less feature
        reduced_features = features_list.drop(features_list[i])
        x_minus = x_train[reduced_features]
        x_valid_minus = x_validation[reduced_features]

        new_model = LogisticRegression(solver='newton-cholesky')
        new_model.fit(X=x_minus, y=y_train)

        # see how well the new model does
        score = new_model.score(x_valid_minus, y_validation)

        if score > best_curr_score:
            best_curr_score = score
            best_model = new_model
            feature_dropped = features_list[i]
            print(f'Feature dropped: {feature_dropped}')
            accuracy_improved = True

    # No improvement found: stop recursion
    if feature_dropped is None:
        print("No more improvement can be found!")

        final_model = LogisticRegression(
            solver='newton-cholesky',
            max_iter=1000
        )

        final_model.fit(x_train[features_list], y_train)

        return final_model, best_score, features_list

    # Remove the best feature found
    print(
        f"Feature dropped: {best_feature_to_drop} "
        f"(score={best_score_this_round:.4f})"
    )

    reduced_features = features_list.drop(best_feature_to_drop)

    # Recurse
    return feature_selection(
        reduced_features,
        best_score_this_round
    )

# MAIN CODE

# randomly assign instances to training, validation, and test sets.
# 80% training, 10% validation, 10% test
random_instances = np.random.random(size=customer_data.shape[0])
train_idx = random_instances < 0.8
validation_idx = (random_instances >= 0.8) & (random_instances < 0.9)
test_idx = random_instances >= 0.9

x_train = customer_data.loc[train_idx]
y_train = customer_claims['claim'].loc[train_idx]

x_validation = customer_data.loc[validation_idx]
y_validation = customer_claims['claim'].loc[validation_idx]

x_test = customer_data.loc[test_idx]
y_test = customer_claims['claim'].loc[test_idx]

# create a supervised learning model
neighbor_model = KNeighborsClassifier(n_neighbors=27).fit(X=x_train, y=y_train)
lr_model = LogisticRegression(solver='newton-cholesky').fit(X=x_train, y=y_train)
print(f'Accuracy with hello neighbor: {neighbor_model.score(x_test, y_test)}')

lr_score = lr_model.score(x_test,y_test)
print(f'Accuracy with logistic regression model: {lr_score}')

# Bidirectional stepwise feature selection. 
features_list = x_train.columns
reduced_model, reduced_model_score, reduced_features = feature_selection(features_list,lr_score)
print(f'Accuracy after feature selection: {reduced_model_score}')
print((len(reduced_features)))
print((reduced_features))

improved = True



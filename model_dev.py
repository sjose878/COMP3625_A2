import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import confusion_matrix

    
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
    feature_dropped = None
    best_curr_score = best_score

    # try every combo of every feature except one
    for i in range(len(features_list)): 

        # create a model with one less feature
        print(f'Feature being tested: {features_list[i]}')
        reduced_features = features_list.drop(features_list[i])
        x_minus = x_train[reduced_features] # CHANGE IF INTERACTION
        x_valid_minus = x_validation[reduced_features]

        new_model = KNeighborsClassifier(n_neighbors=7)
        new_model.fit(X=x_minus, y=y_train)

        # see how well the new model does
        score = new_model.score(x_valid_minus, y_validation)
        if score >= best_curr_score:
            best_curr_score = score
            feature_dropped = features_list[i]

    # No improvement found: stop recursion
    if feature_dropped is None:
        print("No more improvement can be found!")
        # Train reduced model with the best subset of features
        final_model = KNeighborsClassifier(n_neighbors=7)
        final_model.fit(x_train[features_list], y_train) # CHANGE IF INTERACTION

        return final_model, features_list

    # Improvement found: recurse again with worst feature removed
    reduced_features = features_list.drop(feature_dropped)
    print(f"Feature dropped: {feature_dropped}")
    print(f'# of features remaining: {len(reduced_features)}')
    return feature_selection(reduced_features, best_curr_score)

# MAIN CODE

poly = PolynomialFeatures(
    degree=2,
    interaction_only=True,
    include_bias=False
)

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

# Do feature select on random subset. Reduce computation time.
subset_idx = np.random.choice(
    x_train.index,
    size=20000,
    replace=False
)
# The training data to be used in feature selection
x_train_fs = x_train.loc[subset_idx]
y_train_fs = y_train.loc[subset_idx]

# Creat all pairwise interactions of all 21 features
x_train_inter = poly.fit_transform(x_train)
x_validation_inter = poly.transform(x_validation)
x_test_inter = poly.transform(x_test)

# Maintain feature names
feature_names = poly.get_feature_names_out(
    x_train.columns
)

x_train_inter = pd.DataFrame(
    x_train_inter,
    columns=feature_names,
    index=x_train.index
)

x_validation_inter = pd.DataFrame(
    x_validation_inter,
    columns=feature_names,
    index=x_validation.index
)

x_test_inter = pd.DataFrame(
    x_test_inter,
    columns=feature_names,
    index=x_test.index
)

feature_names = x_train_inter.columns

# create a supervised learning model
neighbor_model = KNeighborsClassifier(n_neighbors=9).fit(X=x_train, y=y_train)
lr_model = LogisticRegression(max_iter=1000).fit(X=x_train, y=y_train)
print(f'Accuracy with hello neighbor: {neighbor_model.score(x_test, y_test)}')

lr_score = lr_model.score(x_test,y_test)
print(f'Accuracy with logistic regression model: {lr_score}')

# note to self: features_list -> 21 features, feature_names -> 231 features
features_list = x_train.columns
print((len(feature_names)))
print(features_list)

# Do reverse stepwise feature selection and train a model with subset of features
reduced_model, reduced_features = feature_selection(features_list=features_list, best_score=0)
print(f'Accuracy after feature selection: {reduced_model.score(x_test[reduced_features], y_test)}')
print((len(reduced_features)))
#print((reduced_features))

# Compute accuracy, sensitivity, and specificity of model with confusion matrix
y_predictions = reduced_model.predict(x_test_inter[reduced_features])

tn, fp, fn, tp = confusion_matrix(y_test, y_predictions).ravel()
accuracy = (tp + tn) / (tp + tn + fp + fn)
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

print(f'Accuracy: {accuracy}\nSensitivity: {sensitivity}\nSpecificity: {specificity}')

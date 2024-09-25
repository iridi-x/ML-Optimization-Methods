import numpy as np
import pandas as pd
import time
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


def scale_0_1(x):
    return x / np.max(x)


def import_wine_data_from_csv():
    # wine_data = np.genfromtxt('wine_data.csv', delimiter=',')

    wine_data = pd.read_csv('wine_data.csv', delimiter=';', index_col=False)

    wine_y = wine_data["quality"].to_frame()
    wine_X = wine_data.drop(["quality"], axis=1)

    return wine_X, wine_y


def print_results(results):
        # print results of all models
        print(f"{'Type':<15} {'Degree/Order':<15} {'Training RMSE':<15} {'Training R²':<15} {'Training Time':<15} {'Testing RMSE':<15} {'Testing R²':<15}")
        for result in results:
            print(f"{result['Type']:<15} {result['Order']:<15.0f} {result['Training RMSE']:<15.6f} {result['Training R2']:<15.6f} {result['Training Time']:<15.6f} {result['Testing RMSE']:<15.6f} {result['Testing R2']:<15.6f}")


# Load your data into a DataFrame
# df = pd.read_csv('your_data.csv')  # Example of loading data

# Assuming your features are in X and the target variable is in y
# X = df[['feature1', 'feature2', ...]]
# y = df['target']

def lasso_sklearn(X, y):
    results = []

    for order in range(1, 3): # this tests orders 1-5 

        wine_X_train, wine_X_test, wine_y_train, wine_y_test = train_test_split(X,y, test_size=0.2, random_state=(42+order))

        poly_features = PolynomialFeatures(order) # this uses Sklearn to create polynomial features from dataset
        X_train_polynomial = poly_features.fit_transform(wine_X_train)
        X_test_polynomial = poly_features.transform(wine_X_test)

        # Fit a LASSO model with polynomial features
        start_time = time.time()

        lasso_model = make_pipeline(poly_features, Lasso(alpha=0.0001))  # Set alpha to your initial choice
        lasso_model.fit(X_train_polynomial, wine_y_train)

        # Make predictions
        
        end_time = time.time()

        y_train_pred = lasso_model.predict(X_train_polynomial)
        y_test_pred = lasso_model.predict(X_test_polynomial)

        # Calculate stats
        training_rmse = np.sqrt(mse(wine_y_train, y_train_pred))
        test_rmse = np.sqrt(mse(wine_y_test, y_test_pred))
        train_r2 = r2_score(wine_y_train, y_train_pred)
        test_r2 = r2_score(wine_y_test, y_test_pred)
        training_time = end_time - start_time


        # add data object to results array 
        results.append({
            "Type": "Lasso",
            "Order": order,
            # RMSE measures the distance from prediction to actual, with 0 being no different between predic and actual
            "Training RMSE": training_rmse,
            # R2 is somewhat inverse to RMSE, where a value of 1 indicates no difference between prediction and actual
            "Training R2": train_r2,
            "Training Time": training_time,
            "Testing RMSE": test_rmse,
            "Testing R2":  test_r2
        })

        coefficients = lasso_model.named_steps['lasso'].coef_
        intercept = lasso_model.named_steps['lasso'].intercept_

        # Polynomial terms
        poly_features = poly_features.get_feature_names_out()

        # Combine terms into a readable equation
        terms = [f"{intercept[0]:.2f}"] + [f"{coef:.2f} * {term}" for coef, term in zip(coefficients, poly_features) if coef != 0]
        polynomial_function = " + ".join(terms)

        print("Polynomial function:", polynomial_function)

        # # Identify features with zero coefficients
        # zero_coef_features = [poly_features[i] for i, coef in enumerate(coefficients) if coef == 0]

        # # Identify features with non-zero coefficients (the ones retained by the model)
        # non_zero_coef_features = [poly_features[i] for i, coef in enumerate(coefficients) if coef != 0]

        # print("\n\nFeatures to remove (zero coefficients):", zero_coef_features)
        # print("Features retained:", non_zero_coef_features)

    return results





def main():
    print("main")
    # import data 
    wine_X, wine_y = import_wine_data_from_csv()

    # preprocess data
    # print(f"Preprocessed: \n {wine_X[:1]}")
    wine_X = scale_0_1(wine_X)
    # print(f"Processed: \n {wine_X[:1]} \n\n")


    results = lasso_sklearn(wine_X, wine_y)
    print_results(results)


if __name__ == '__main__':
    main()
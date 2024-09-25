import numpy as np
import pandas as pd
import time

from sklearn.metrics import mean_squared_error as mse, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, Normalizer
from sklearn.linear_model import LinearRegression


# store resuslts of each model
results = []


# ======================================
def min_max_scaling(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def scale_0_1(x):
    return x / np.max(x)



def scale_minus1_1(x):
    return 2 * (x - np.min(x)) / (np.max(x) - np.min(x)) - 1


def z_score_normalization(x):
    return (x - np.mean(x)) / np.std(x)


def mean_normalization(x):
    return (x - np.mean(x)) / (np.max(x) - np.min(x))


# ======================================

# scale data
def scale_data(data):
    print("process data")

    print("original Data: \n", data[:1])
    # scale the data
    scalar = StandardScaler()
    data_X = scalar.fit_transform(data)

    print("Processed Data:\n", data_X[:1])

    return data_X


# normailize data
def normalize_data(data):
    print("process data")

    print("original Data: \n", data[:1])
    # normalize the data
    normal = Normalizer(norm='l2')
    data_X = normal.fit_transform(data)

    print("Processed Data:\n", data_X[:1])

    return data_X



def import_wine_data_from_csv():
    # wine_data = np.genfromtxt('wine_data.csv', delimiter=',')

    wine_data = pd.read_csv('wine_data.csv', delimiter=';', index_col=False)

    wine_y = wine_data["quality"].to_frame()
    wine_X = wine_data.drop(["quality"], axis=1)

    return wine_X, wine_y


def calculate_w(data_X, data_y):
    # print("Manual Weights:")
    X = np.array(data_X)
    y = np.array(data_y)
    X_t = X.T
    X_t_X = np.dot(X_t, X)
    X_t_y = np.dot(X_t, y)
    X_t_X_inv = np.linalg.inv(X_t_X)
    w_hat =  np.dot(X_t_X_inv, X_t_y)
    # print("Weights: \n", w_hat)
    return w_hat


def least_square_solution(train_data_X, train_data_y, X_new):
   

    X = np.array(train_data_X)
    y = np.array(train_data_y)
    X_t = X.T
    X_t_X = np.dot(X_t, X)
    X_t_y = np.dot(X_t, y)
    X_t_X_inv = np.linalg.inv(X_t_X)

    # w_hat represents the weight vector
    w_hat =  np.dot(X_t_X_inv, X_t_y)
    X_new = np.array(X_new)
    w_hat_t = w_hat.T

    # this one seems to produce similar results, not sure why as it does not follow the formula in the notes
    # y_hat = np.dot(X_new, w_hat)

    y_hat = []
    for row in X_new:
        y_i = np.dot(w_hat_t, row)
        y_hat.append(y_i)

    y_hat = np.array(y_hat)

    return y_hat

def print_results(results):
        # print results of all models
        print(f"{'Type':<15} {'Degree/Order':<15} {'Training RMSE':<15} {'Training R²':<15} {'Training Time':<15} {'Testing RMSE':<15} {'Testing R²':<15}")
        for result in results:
            print(f"{result['Type']:<15} {result['Order']:<15.0f} {result['Training RMSE']:<15.6f} {result['Training R2']:<15.6f} {result['Training Time']:<15.6f} {result['Testing RMSE']:<15.6f} {result['Testing R2']:<15.6f}")



def least_square_method_np(wine_X, wine_y):

    for order in range(1, 6): # this tests orders 1-5 

        wine_X_train, wine_X_test, wine_y_train, wine_y_test = train_test_split(wine_X,wine_y, test_size=0.2, random_state=(42+order))

        poly = PolynomialFeatures(order) # this uses Sklearn to create polynomial features from dataset
        X_train_polynomial = poly.fit_transform(wine_X_train)
        X_test_polynomial = poly.transform(wine_X_test)

        # train and benchmark the model
        model = LinearRegression()

        # start timer
        start_time = time.time()

        # train model
        model.fit(X_train_polynomial, wine_y_train)

        # end timer
        end_time = time.time()

        # test model 
        y_train_pred = model.predict(X_train_polynomial)
        y_test_pred  = model.predict(X_test_polynomial)

        # get performance metrics
        training_rmse = np.sqrt(mse(wine_y_train, y_train_pred))
        test_rmse     = np.sqrt(mse(wine_y_test, y_test_pred))
        train_r2      = r2_score(wine_y_train, y_train_pred) 
        test_r2       = r2_score(wine_y_test, y_test_pred)
        training_time = (end_time - start_time)





        # add data object to results array 
        results.append({
            "Type": "Sklearn",
            "Order": order,
            # RMSE measures the distance from prediction to actual, with 0 being no different between predic and actual
            "Training RMSE": training_rmse,
            # R2 is somewhat inverse to RMSE, where a value of 1 indicates no difference between prediction and actual
            "Training R2": train_r2,
            "Training Time": training_time,
            "Testing RMSE": test_rmse,
            "Testing R2":  test_r2
        })

        # for our selected model
        if order == 2:
            print("order:", order)
            # print("Intercept: \n", intercept,"\n Weights: \n", weights)


            # manual_w = calculate_w(X_train_polynomial, wine_y_train)
            # print(manual_w.T)

            # treain manual model
            man_start_time = time.time()

            train_prediction = least_square_solution(X_train_polynomial, wine_y_train,  X_train_polynomial)
            test_prediction = least_square_solution(X_train_polynomial, wine_y_train,  X_test_polynomial)

            man_end_time = time.time()

            # stats
            man_training_rmse = np.sqrt(mse(wine_y_train, train_prediction))
            man_train_r2      = r2_score(wine_y_train, train_prediction) 

            man_test_rmse     = np.sqrt(mse(wine_y_test, test_prediction))
            man_test_r2       = r2_score(wine_y_test, test_prediction)

            man_training_time = (man_end_time - man_start_time)

            # append stats
            results.append({
                "Type": "manual",
                "Order": order,
                # RMSE measures the distance from prediction to actual, with 0 being no different between predic and actual
                "Training RMSE": man_training_rmse,
                # R2 is somewhat inverse to RMSE, where a value of 1 indicates no difference between prediction and actual
                "Training R2": man_train_r2,
                "Training Time": man_training_time,
                "Testing RMSE": man_test_rmse,
                "Testing R2":  man_test_r2
            })

            num_features = wine_X.shape[1]
            coefficients = model.coef_.flatten()
            intercept = model.intercept_

            # Combine terms into a readable equation
            terms = [f"{intercept[0]:.2}"]  # Use intercept directly as a float


            # terms = [f"{intercept[0]:.2f}"] + [f"{coef:.2f} * {term}" for coef, term in zip(coefficients, poly_features) if coef != 0]
            for index, coef in enumerate(coefficients):
                    # Check if coefficient is not zero and if it's a linear (degree 1) or quadratic (degree 2) term
                if coef != 0.0 and (0 <= index < 2 * num_features + 1):
                    terms.append(f"{coef:.2f} * x{index + 1}")  # Use x0 for intercept, x1, x2,... for features


            # Join all terms to create the polynomial function string
            polynomial_function = " + ".join(terms)

            print("Polynomial function:", polynomial_function)
                
    return results




def main():
    print("main")
    # import data 
    wine_X, wine_y = import_wine_data_from_csv()

 

    print(f"Preprocessed: \n {wine_X[:1]}")

    # preprocess data - scaling only columns with numbers greater than little to no effect
    wine_X = scale_0_1(wine_X)


    # scale only columns whose max are greater than one (this is for numpy arrays)
    # for column in range(wine_X.shape[1]):
    #     if np.max(wine_X[:, column]) > 1:
    #         wine_X[:,column] = scale_0_1(wine_X[:, column])

    # # for pd.df
    # for col in wine_X.columns:
    #     if wine_X[col].max() >1:
    #         wine_X[col] = scale_0_1(wine_X[col])

    # wine_y = scale_0_1(wine_y)
    print(f"Processed: \n {wine_X[:1]} \n\n")

    
    print(f"Min: {np.min(wine_y)}, Max: {np.max(wine_y)}, Mean: {np.mean(wine_y)}")



    results = least_square_method_np(wine_X, wine_y)    
    print_results(results)


if __name__ == '__main__':
    main()
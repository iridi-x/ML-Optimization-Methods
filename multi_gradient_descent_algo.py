import numpy as np
import pandas as pd
import time

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error as mse, r2_score
from sklearn.model_selection import train_test_split





def min_max_scaling(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def scale_0_1(x):
    return x / np.max(x)


def mean_normalization(x):
    return (x - np.mean(x)) / (np.max(x) - np.min(x))


def import_wine_data_from_csv():
    # wine_data = np.genfromtxt('wine_data.csv', delimiter=',')

    wine_data = pd.read_csv('wine_data.csv', delimiter=';', index_col=False)

    wine_y = wine_data["quality"].to_frame()
    wine_X = wine_data.drop(["quality"], axis=1)

    return wine_X, wine_y




def print_polynomial_function(weights):
    # Prints the polynomial function based on the learned weights.
    terms = []
    for i, weight in enumerate(weights):
        if weight != 0:
            terms.append(f"{weight:.4f} * x{i}" if i > 0 else f"{weight:.4f}")
    
    polynomial_function = " + ".join(terms)
    print("Polynomial Function: f(x) =", polynomial_function)




def print_results(results):
        # print results of all models
        print(f"{'Type':<15} {'Degree/Order':<15} {'Training RMSE':<15} {'Training R²':<15} {'Training Time':<15} {'Testing RMSE':<15} {'Testing R²':<15}")
        for result in results:
            print(f"{result['Type']:<15} {result['Order']:<15} {result['Training RMSE']:<15.6f} {result['Training R2']:<15.6f} {result['Training Time']:<15.6f} {result['Testing RMSE']:<15.6f} {result['Testing R2']:<15.6f}")


# ======================================

# with this method, a higher alpha value appears to produce more accurate results, this is opposite of LASSO
def gradient_descent(data_X, data_y, iterations=500, alpha=2, batch_size=30):
    print("multi- feature gradient descent")
    results = []

    # split the data for training and testing
    data_X_train, data_X_test, data_y_train, data_y_test = train_test_split(data_X,data_y, test_size=0.2)

    # convert input to np array
    data_X_train = np.array(data_X_train)
    data_y_train = np.array(data_y_train)
    data_X_test = np.array(data_X_test)
    data_y_test = np.array(data_y_test)

    # set data info
    num_samples, num_features = data_X_train.shape

    # initalize weight to a random variable 
    weight = np.random.randn(num_features)

    # variables to collect best weights
    best_weight = weight.copy()
    best_mse = np.inf

    # start timer
    start_time = time.time()

    for i in range(iterations):


        # randomize data samples
        indices = np.random.permutation(num_samples)
        X_shuffled = data_X_train[indices]
        y_shuffled = data_y_train[indices]

        # begin the descent
        for j in range(0, num_samples, batch_size):
            end = j + batch_size
            X_batch = X_shuffled[j:end]

            # need to flatten this otherwise the weight update will fail
            y_batch = y_shuffled[j:end].ravel()

            # predict with current weights
            y_predic = np.dot(X_batch, weight)

            # calculate the gradient
            gradient = (-2/len(y_batch)) * np.dot(X_batch.T, (y_batch - y_predic))

            a_g = alpha * gradient

            # update the weight
            weight = weight - (a_g)

            batch_mse = mse(y_batch, y_predic)

            if batch_mse < best_mse:
                best_mse = batch_mse
                best_weight = weight.copy()

        if i % 50 == 0:
            print(f'iteration {i}: batch_MSE = {batch_mse}')

    # END timer
    end_time = time.time()

    # print best mse and weights
    print(f'best_mse = {best_mse}')

    # test resulting model
    y_train_pred = np.dot(data_X_train, best_weight)
    y_test_pred = np.dot(data_X_test, best_weight)

    # calculate stats

            # get performance metrics
    training_rmse = np.sqrt(mse(data_y_train, y_train_pred))
    test_rmse = np.sqrt(mse(data_y_test, y_test_pred))
    train_r2 = r2_score(data_y_train, y_train_pred)
    test_r2 = r2_score(data_y_test, y_test_pred)
    training_time = end_time - start_time



    results.append({
    "Type": "Batch",
    "Order": "Linear",
    # RMSE measures the distance from prediction to actual, with 0 being no different between predic and actual
    "Training RMSE": training_rmse,
    # R2 is somewhat inverse to RMSE, where a value of 1 indicates no difference between prediction and actual
    "Training R2": train_r2,
    "Training Time": training_time,
    "Testing RMSE": test_rmse,
    "Testing R2":  test_r2
    })

    # print the polynomial function
    print_polynomial_function(weight)



    # ==================================================
    return results




def main():
    print("main")
    # import data 
    wine_X, wine_y = import_wine_data_from_csv()

    # preprocess data
    # print(f"Preprocessed: \n {wine_X[:1]}")
    wine_X = scale_0_1(wine_X)
    # print(f"Processed: \n {wine_X[:1]} \n\n")


    results = gradient_descent(wine_X, wine_y, iterations=1000)    
    print_results(results)


if __name__ == '__main__':
    main()
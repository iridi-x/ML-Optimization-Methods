import numpy as np
import pandas as pd
import time

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error as mse, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
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

# results object 
            # results.append({
            #     "Type": "manual",
            #     "Order": order.degree,
            #     # RMSE measures the distance from prediction to actual, with 0 being no different between predic and actual
            #     "Training RMSE": man_training_rmse,
            #     # R2 is somewhat inverse to RMSE, where a value of 1 indicates no difference between prediction and actual
            #     "Training R2": man_train_r2,
            #     "Training Time": man_training_time,
            #     "Testing RMSE": man_test_rmse,
            #     "Testing R2":  man_test_r2
            # })

# ======================================

def mean_squared_error(y_true, y_predicted):
     
    # Calculating the loss or cost
    cost = np.sum((y_true-y_predicted)**2) / len(y_true)
    return cost

# Cost function: Mean Squared Error (MSE)
def cost_mse(X, y, theta):
    m = len(y)
    return (1/m) * np.sum((np.dot(X, theta) - y) ** 2)

def gradient_descent(data_X, data_y, iterations, learning_rate = 0.0001, stopping_threshold = 1e-6):
    print("gradient descent")
    # initalize w_0 to a random variable 
    weight = 0.1
    bias = 0.01
    # set hyper parameter/theta/learning rate 
    learning_rate = learning_rate
    iterations = iterations
    n = float(len(data_X))

    costs = []
    weights = []
    previous_cost = None


    for i in range(iterations):
        # make prediction
        y_predic = (weight * data_X) + bias

        # calculate weight
        current_cost = mean_squared_error(data_y, y_predic)

        # check the change in cost and break if cahnged (alter this later)
        if previous_cost and abs(previous_cost-current_cost)<=stopping_threshold:
            break
         
        previous_cost = current_cost

        costs.append(current_cost)
        weights.append(current_weight)

        # Calculate the gradients, take 1st d/dx
        weight_derivative = -(2/n) * sum(x * (y-y_predic))
        bias_derivative = -(2/n) * sum(y-y_predic)
         

        current_weight = current_weight - (learning_rate * weight_derivative)
        current_bias = current_bias - (learning_rate * bias_derivative)
                 


    return current_weight, current_bias




def main():
    print("main")
    # import data 
    wine_X, wine_y = import_wine_data_from_csv()

    # preprocess data
    # print(f"Preprocessed: \n {wine_X[:1]}")
    wine_X = scale_0_1(wine_X)
    # print(f"Processed: \n {wine_X[:1]} \n\n")


    estimated_weight, estimated_bias = gradient_descent(wine_X, wine_y, iterations=5)    
    print_results(results)


if __name__ == '__main__':
    main()
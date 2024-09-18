import matplotlib.pyplot as plt
import numpy as np
import time

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error as mse, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression

# store resuslts of each model
results = []

# normailize data
def normalize_data(data):
    print("process data")

    print("original Data:", data[2:])
    scalar = StandardScaler()
    # normalize the data
    data_X = scalar.fit_transform(data)

    print("Processed Data:", data_X[2:])

    return data_X




# EXAMPLE FROM LECTURE SLIDES 
def regressionExampleDiabetes():
    print("least square solution")

    # load the dataset
    diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

    # use only one feature
    diabetes_X = diabetes_X[:, np.newaxis, 2]

    # split the data into train and test    
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test  = diabetes_X[-20:]

    # split targets into train and test
    diabetes_y_train = diabetes_y[:-20] 
    diabetes_y_test  = diabetes_y[-20:] 

    # create linear regression obj
    lregr = linear_model.LinearRegression()

    # train the model
    lregr.fit(diabetes_X_train, diabetes_y_train)

    # make predictions using the testing set 
    diabetes_y_predic = lregr.predict(diabetes_X_test)
    print("coeffcients \n", lregr.coef_)

    # print the MSE (mean standard error)
    print("Mean squared error: %.2f" % mse(diabetes_y_test, diabetes_y_predic))

    # the coef of determination: (1 is a perfect score)
    print("Coefficient of determination: %.2f" % r2_score(diabetes_y_test, diabetes_y_predic))

    # plot outputs 
    plt.scatter(diabetes_X_test, diabetes_y_test, color="black")
    plt.plot(diabetes_X_test, diabetes_y_predic, color="blue", linewidth=3)
    plt.xticks(())
    plt.yticks(())
    plt.show()




def least_square_method_np():

    # load the dataset
    wine_X, wine_y = datasets.load_wine(return_X_y=True)

    # do we need to split features for this one? 

    # 80 - 20 split, training to testing
    wine_X_train, wine_X_test, wine_y_train, wine_y_test = train_test_split(wine_X,wine_y, test_size=0.2, random_state=42)

   
    for order in range(1, 6): # this tests orders 1-5 

        order = PolynomialFeatures(order) # this uses Sklearn to create polynomial features from dataset
        X_train_polynomial = order.fit_transform(wine_X_train)
        # X_train_standardized = normalize_data(X_train_polynomial)
        X_test_polynomial = order.transform(wine_X_test)

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

        # get perfimance metrics
        training_rmse = np.sqrt(mse(wine_y_train, y_train_pred))
        test_rmse     = np.sqrt(mse(wine_y_test, y_test_pred))
        train_r2      = r2_score(wine_y_train, y_train_pred) 
        test_r2       = r2_score(wine_y_test, y_test_pred)
        training_time = (end_time - start_time)

        # add data object to results array 
        results.append({
            "Order": order.degree,
            # RMSE measures the distance from prediction to actual, with 0 being no different between predic and actual
            "Training RMSE": training_rmse,
            # R2 is somewhat inverse to RMSE, where a value of 1 indicates no difference between prediction and actual
            "Training R2": train_r2,
            "Training Time": training_time,
            "Testing RMSE": test_rmse,
            "Testing R2":  test_r2
        })
 
    # print results of all models
    print(f"{'Degree/Order':<15} {'Training RMSE':<15} {'Training R²':<15} {'Training Time':<15} {'Testing RMSE':<15} {'Testing R²':<15}")
    for result in results:
        print(f"{result['Order']:<15.0f} {result['Training RMSE']:<15.6f} {result['Training R2']:<15.6f} {result['Training Time']:<15.6f} {result['Testing RMSE']:<15.6f} {result['Testing R2']:<15.6f}")

   
   
    # create model - we need to split the data and pass it now 
    # lsm_np = np.linalg.lstsq()



def main():
    print("main")
    # regressionExampleDiabetes()
    least_square_method_np()



if __name__ == '__main__':
    main()
import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error as mse, r2_score




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

    # create model
    lsm_np = np.linalg.lstsq()






def main():
    print("main")
    regressionExampleDiabetes()



if __name__ == '__main__':
    main()
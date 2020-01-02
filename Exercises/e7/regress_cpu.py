import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pykalman import KalmanFilter
from sklearn.linear_model import LinearRegression


X_columns = ['temperature', 'cpu_percent', 'fan_rpm', 'sys_load_1']
y_column = 'next_temp'


def get_data(filename):
    """
    Read the given CSV file. Returns sysinfo DataFrames for training and testing.
    """
    sysinfo = pd.read_csv(filename, parse_dates=['timestamp'])
    
    
    sysinfo[y_column] = sysinfo['temperature'].shift(-1)            #temperature value from the next row
    sysinfo = sysinfo[sysinfo[y_column].notnull()]                  #the last row should have y_column null: no next temp known
    
    # return most of the data to train, and pick an interesting segment to test
    split1 = int(sysinfo.shape[0] * 0.8)
    split2 = int(sysinfo.shape[0] * 0.84)
    train = sysinfo.iloc[:split1, :]
    test = sysinfo.iloc[split1:split2, :]
    
    return train, test


def get_trained_coefficients(X_train, y_train):
    """
    Create and train a model based on the training_data_file data.

    Return the model, and the list of coefficients for the 'X_columns' variables in the regression.
    """
    
    model = LinearRegression(fit_intercept=False)       #create linear regression model
    model.fit(X_train, y_train)                         #train the model with training data
    
    coefficients = model.coef_                      #returns coefficients of the eqn...in this case it is intercept and slope since it is linear eqn

    return model, coefficients


def output_regression(coefficients):
    regress = ' + '.join('%.3g*%s' % (coef, col) for col, coef in zip(X_columns, coefficients))
    print('next_temp = ' + regress)


def plot_errors(model, X_test, y_test):
    plt.hist(y_test - model.predict(X_test), bins=100)
    plt.savefig('test_errors.png')


def smooth_test(coef, sysinfo):
    X_test, y_test = sysinfo[X_columns], sysinfo[y_column]
    
    # feel free to tweak these if you think it helps.
    transition_stddev = 1.5
    observation_stddev = 2.0

    dims = X_test.shape[-1]
    initial = X_test.iloc[0]
    observation_covariance = np.diag([observation_stddev, 2, 10, 1]) ** 2
    transition_covariance = np.diag([transition_stddev, 80, 100, 10]) ** 2
    
    # Transition = identity for all variables, except we'll replace the top row
    # to make a better prediction, which was the point of all this.
    transition = np.eye(dims) # identity matrix, except...
    
    transition = [[coef[0], coef[1], coef[2], coef[3]], [0, 0.6, 0, 0.03], [0, 0, 0, 0], [0, 1.3, 0, 0.8]]      #use prediction values from the linear regression


    kf = KalmanFilter(
        initial_state_mean=initial,
        initial_state_covariance=observation_covariance,
        observation_covariance=observation_covariance,
        transition_covariance=transition_covariance,
        transition_matrices=transition,
    )

    kalman_smoothed, _ = kf.smooth(X_test)

    plt.figure(figsize=(15, 6))
    plt.plot(sysinfo['timestamp'], sysinfo['temperature'], 'b.', alpha=0.5)
    plt.plot(sysinfo['timestamp'], kalman_smoothed[:, 0], 'g-')
    plt.savefig('smoothed.png')


def main():
    train, test = get_data(sys.argv[1])
    X_train, y_train = train[X_columns], train[y_column]
    X_test, y_test = test[X_columns], test[y_column]

    model, coefficients = get_trained_coefficients(X_train, y_train)
    output_regression(coefficients)
    #print("Training score: %g\nTesting score: %g" % (model.score(X_train, y_train), model.score(X_test, y_test)))

    plot_errors(model, X_test, y_test)
    smooth_test(coefficients, test)


if __name__ == '__main__':
    main()

# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression

def data_extract():
    df = pd.read_csv(r'Data\Data1.csv')
    data_frame = df[["T", "P", "TC", "SV"]]
    print(data_frame.shape)
    return data_frame


def mean(d_set):
    mean_dict = dict()
    for col in d_set:
        mean_value = np.mean(d_set[col].values)
        mean_dict[col] = mean_value
    # print("printing mean values ", mean_dict)
    return mean_dict


def data_standardization(dt_set):
    stan_data = pd.DataFrame()
    mean_values = mean(dt_set)
    for column in dt_set:
        stan_data[column] = dt_set[column] - mean_values[column]

    # print("printing first 10 values of standard data\n", stan_data.head(10))
    return stan_data


def variance(d_set):
    variance_dict = dict()
    for col in d_set:
        variance_value = np.var(d_set[col].values)
        variance_dict[col] = variance_value
    # print("printing variance values ", variance_dict)
    return variance_dict


def standard_deviation(dt_set):
    var_value = variance(dt_set)
    s_value = dict()
    for key in var_value:
        s_value[key] = np.sqrt(var_value[key])

    print("printing standard deviation values\n", s_value)


def covariance_matrix(dataset):
    mean_values = mean(dataset)
    variance_values = variance(dataset)
    covariance = dict()
    length = len(dataset)
    # no_of_columns = len(dataset.columns)

    """covariance = np.sum((dataset["T"].values - np.mean(dataset["T"].values)) * (dataset["P"].values - np.mean(dataset["P"].values))) / (length - 1)
    print("covariance of T and P columns is ", covariance)"""

    for column1 in dataset.iloc[:, dataset.columns != "SV"]:
        for column2 in dataset.columns[::-1]:
            if column2 != column1:
                cv = (np.sum((dataset[column1].values - mean_values[column1]) * (dataset[column2].values - mean_values[column2]))) / (length - 1)
                # print("covariance of %s, and %s is %s " % (column1, column2, cv))
                covariance[column1, column2] = cv
            else:
                break

    print("covariance values are ", covariance)
    cov_matrix = np.mat([[variance_values['T'], covariance[('T', 'P')], covariance[('T', 'TC')], covariance[('T', 'SV')]],
                           [covariance[('T', 'P')], variance_values['P'], covariance[('P', 'TC')], covariance[('P', 'SV')]],
                           [covariance[("T", "TC")], covariance[("P", "TC")], variance_values["TC"], covariance[("TC", "SV")]],
                           [covariance[("T", "SV")], covariance[("P", "SV")], covariance[("TC", "SV")], variance_values["SV"]]
                           ])

    print("printing covariance matrix")
    # print(cov_matrix)
    return cov_matrix


def find_eigen(mat):
    eigen_values, eigen_vectors = np.linalg.eig(mat)
    print("printing eigen values \n", eigen_values)
    print("printing eigen vectors \n", eigen_vectors)
    return eigen_values, eigen_vectors


def projection_matrix(data_set, eig):
    eigen_vector = np.transpose(eig)
    # print("printing eigen vector transpose\n", eigen_vector)
    pc_matrix = np.matmul(data_set, eigen_vector)
    print("printing pc_matrix size ", pc_matrix.shape)
    pc_array = np.array(pc_matrix)
    p_matrix = np.delete(pc_array, np.s_[2:], 1)
    print("printing the shape of project Matrix ", p_matrix.shape)
    # print("printing first 10 rows of pc_array\n", pc_array[:10, :])
    # print("printing first 10 rows of projection matrix\n", project_matrix[:10, :])


def explained_variance(eigen_value):
    print("eigen values are ", eigen_value)
    sum_value = 0
    i = 1
    ex_variance = dict()
    for value in eigen_value:
        sum_value = sum_value + value

    # print("total sum of eigen values is ", sum_value)
    for var in eigen_value:
        percentage = (var/sum_value) * 100
        key = "PC" + str(i)
        # print("percentage of %s is %s" % (key, percentage))
        ex_variance[key] = percentage
        i = i + 1

    print("printing explained variance for each PC ", ex_variance)


def component_matrix(vector):
    print("printing eigen vector\n", vector)
    e_vec = np.transpose(vector)
    print("printing transpose vector\n", e_vec)
    column_names = ["PC1", "PC2", "PC3", "PC4"]
    row_names = ["T", "P", "TC", "SV"]
    c_frame = pd.DataFrame(e_vec, columns=column_names, index=row_names)
    print("component data frame\n", c_frame)

def kaiserCriteria(eigenValues):
    eigenAvg = np.average(eigenValues)
    print('Avg eigen value : {}'.format(eigenAvg))
    print('Eigen value greater than Avg : ')
    print([val for val in eigenValues if val > eigenAvg])

def pc_score(data, eigenVector):
    return np.matmul(data.iloc[:,0:4], eigenVector)

def correlation_matrix(data, pcScore):
    ## correlation(x1,x2) = cov(x1,x2)/(s1*s2)
    full_matrix = pd.concat([data, pcScore], axis=1)
    corr_matrix = full_matrix.corr().round(2)
    sns.heatmap(corr_matrix, annot = True, vmax = 1, vmin = -1, center=0, cmap='vlag')
    plt.show()
    
def lasso(data):
    X = data.iloc[:, :-1].values
    Y = data.iloc[:, -1].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    reg = LinearRegression().fit(X_train, Y_train)
    print('Linear Reg score test : {}'.format(reg.score(X_test, Y_test)))
    print('Linear Reg score train: {}'.format(reg.score(X_train, Y_train)))

    lasso = Lasso(alpha=50, max_iter=100, tol=0.1)
    lasso.fit(X_train, Y_train)
    
    print('Lasso Reg score test : {}'.format(lasso.score(X_test, Y_test)))
    print('Lasso Reg score train: {}'.format(lasso.score(X_train, Y_train)))
    print('Lasso Coefficients : {}'.format(lasso.coef_))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data = data_extract()
    s_data = data_standardization(data)
    m_cov = covariance_matrix(s_data)
    eigen_val, eigen_vec = find_eigen(m_cov)
    explained_variance(eigen_val)
    projection_matrix(s_data, eigen_vec)
    component_matrix(eigen_vec)
    standard_deviation(s_data)
    kaiserCriteria(eigen_val)

    pcScore = pc_score(s_data, eigen_vec)
    lasso(s_data)
    correlation_matrix(s_data, pcScore)
    
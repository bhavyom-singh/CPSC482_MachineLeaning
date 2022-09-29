import pandas as p
import numpy as np
import logging as log

log.basicConfig(format='%(asctime)s %(message)s',datefmt='%Y-%m-%d %H:%M:%S', level=log.INFO)

all_data = p.read_csv('Data\Data1.csv')
all_data['constant'] = 1
cols = all_data.columns.tolist()
cols = cols[-1:] + cols[:-1]
all_data = all_data[cols]

data_row_count = len(all_data)

#print(all_data.head(10))
print('Enter the polynomial degree, or 0 to exit')
degree = int(input())
print('Enter training data size(total data size = {})'.format(data_row_count))
training_data_size = int(input())
if degree != 0:
    #print(len(all_data))
    log.info('Training Started')
    mat_normal = np.power(all_data.iloc[0:data_row_count,0:5].values, degree)
    #print(mat_normal[0:10])
    mat_transpose = mat_normal.transpose()
    mat_inverseMul = np.linalg.inv(np.matmul(mat_transpose,mat_normal))
    mat_y = np.power(all_data.iloc[0:data_row_count,5].values,1)
    #print(mat_y)
    w_new = (np.matmul(np.matmul(mat_inverseMul,mat_transpose), mat_y)).transpose()
    #print(w_new)
    #print(all_data.head(10))
    print(f'y = {w_new[0]} + {w_new[1]} * x1 + {w_new[2]} * x2 + {w_new[3]} * x3 + {w_new[4]} * x4')
    log.info('Training Stopped')

'''Residual Sum of Square(RSS)'''
'''RSS = SUM((y(actual) - y(pridicted))sq)'''
def calculate_RSS():
    training_RSS = 0
    testing_RSS = 0
    for idx, row in all_data.iterrows():
        if idx < training_data_size:
            training_RSS += (row['Idx'] - (w_new[0] + w_new[1] * row['T'] + w_new[2] * row['P'] + w_new[3] * row['TC'] + w_new[4] * row['SV']))**2
        if idx >= training_data_size:
            testing_RSS += (row['Idx'] - (w_new[0] + w_new[1] * row['T'] + w_new[2] * row['P'] + w_new[3] * row['TC'] + w_new[4] * row['SV']))**2

    print(f'Training RSS = {training_RSS} , Testing RSS = {testing_RSS}')
    return training_RSS, testing_RSS
''' RSS end'''

''' Mean Squared Error '''
''' MSE = (1/N)RSS '''
def calculate_MSE(training_RSS, testing_RSS):
    training_MSE = (1/data_row_count)*training_RSS
    testing_MSE = (1/data_row_count)*testing_RSS

    print(f'Training MSE = {training_MSE} , Testing MSE = {testing_MSE}')
    return training_MSE, testing_MSE
'''MSE end '''

''' Root MSE '''
''' RMSE = sqrt(MSE)'''
def calculate_RMSE(training_MSE, testing_MSE):
    training_RMSE = np.sqrt(training_MSE)
    testing_RMSE = np.sqrt(testing_MSE)

    print(f'Training RMSE = {training_RMSE} , Testing RMSE = {testing_RMSE}')

''' RMSE end'''

'''R**2, Coefficient of Determination'''
'''R**2 = 1-(RSS/TSS)'''
'''TSS = SUM(yactual - ypridicted)**2'''
def calculate_R2(training_RSS, testing_RSS):
    TSS_training = 0
    TSS_testing = 0
    y_new_sum_training = 0
    y_new_sum_testing = 0
    for idx, row in all_data.iterrows():
        if idx < training_data_size:
            y_new_sum_training += w_new[0] + w_new[1] * row['T'] + w_new[2] * row['P'] + w_new[3] * row['TC'] + w_new[4] * row['SV']
        else:
            y_new_sum_testing += w_new[0] + w_new[1] * row['T'] + w_new[2] * row['P'] + w_new[3] * row['TC'] + w_new[4] * row['SV']

    y_mean_training = y_new_sum_training/(training_data_size)
    y_mean_testing = y_new_sum_testing/(data_row_count-training_data_size)
    
    for idx, row in all_data.iterrows():
        if idx < training_data_size:
            TSS_training += np.square(row['Idx'] - y_mean_training)
        else:
            TSS_testing += np.square(row['Idx'] - y_mean_testing)
    
    print(f'Trainingg TSS = {TSS_training} , Testing TSS = {TSS_testing}')

    training_R2 = 1 - (training_RSS/TSS_training)
    testing_R2 = 1 - (testing_RSS/TSS_testing)

    print(f'Training R-squared = {training_R2} , Testing R-squared = {testing_R2}')
'''R**2 end'''

if __name__ == '__main__':
    training_rss,testing_rss = calculate_RSS()
    training_mse, testing_mse = calculate_MSE(training_rss,testing_rss)
    calculate_RMSE(training_mse, testing_mse)
    calculate_R2(training_rss,testing_rss)

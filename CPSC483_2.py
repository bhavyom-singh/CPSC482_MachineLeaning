import pandas as p
import numpy as np


all_data = p.read_csv('Data\Data1.csv')
#all_data['constant'] = 1
#cols = all_data.columns.tolist()
#cols = cols[-1:] + cols[:-1]
#all_data = all_data[cols]
data_row_count = len(all_data)

print('Enter the polynomial degree, or 0 to exit')
degree = int(input())
df = all_data.drop('Idx', axis=1)
df_norm = np.power(df, degree)
all_data = p.concat((df_norm, all_data.Idx), 1)

print('Enter training data size(total data size = {})'.format(data_row_count))
training_data_size = int(input())

w0 = w1 = w2 = w3 = w4 = 0

def y_predicted_mean(w0,w1,w2,w3,w4):
    y_new_training = 0
    mean = 0
    y_new_testing = 0
    for idx, row in all_data.iterrows():
        if idx < training_data_size:
            y_new_training = w0 + w1*row['T'] + w2*row['P'] + w3*row['TC'] + w4*row['SV']
        else:
            y_new_testing = w0 + w1*row['T'] + w2*row['P'] + w3*row['TC'] + w4*row['SV']

    y_mean_training = y_new_training/(training_data_size)
    y_mean_testing = y_new_testing/(data_row_count-training_data_size)
    return y_mean_training, y_mean_testing

#print('Enter alpha')
#alpha = float(input())
alpha = 0.1983
loop_run = 20
total_row = len(all_data)
inv_total_row = 1/total_row
#print(inv_total_row)
for x in range(loop_run):
    y_new = w0 + w1 * all_data['T'][x] + w2 * all_data['P'][x] + w3 * all_data['TC'][x] + w4 * all_data['SV'][x]
    
    w0 = w0 - alpha * inv_total_row*sum([y_new - all_data['Idx'][x]])
    w1 = w1 - alpha * inv_total_row*sum([all_data['T'][x]*(y_new - all_data['Idx'][x])])
    w2 = w2 - alpha * inv_total_row*sum([all_data['P'][x]*(y_new - all_data['Idx'][x])])
    w3 = w3 - alpha * inv_total_row*sum([all_data['TC'][x]*(y_new - all_data['Idx'][x])])
    w4 = w4 - alpha * inv_total_row*sum([all_data['SV'][x]*(y_new - all_data['Idx'][x])])

    mse = inv_total_row* sum([val**2 for val in [(all_data['Idx'][x] - y_new)]])
    rmse = np.sqrt(mse)
    rss = mse * len(all_data)
    y_mean_training, y_mean_testing = y_predicted_mean(w0,w1,w2,w3,w4)
    tss_training = sum([val**2 for val in [(all_data['Idx'][x] - y_mean_training)]])
    tss_testing = sum([val**2 for val in [(all_data['Idx'][x] - y_mean_testing)]])
    
    r2_training = 1-(rss/(tss_training))
    r2_testing = 1-(rss/(tss_testing))
    print('RSS {}, MSE {}, RMSE {}, R2 training {}, R2 testing {}, w0 {}, w1 {}, w2 {}, w3 {}, w4 {} , loop no {}'.format(rss,mse, rmse, r2_training, r2_testing,w0,w1,w2,w3,w4,x))


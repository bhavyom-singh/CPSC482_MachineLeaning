import pandas as p
import numpy as np
import logging as log

log.basicConfig(format='%(asctime)s %(message)s',datefmt='%Y-%m-%d %H:%M:%S', level=log.INFO)

all_data = p.read_csv('Data\Data1.csv')

log.info('Scaling Started')
print('original data')
print(all_data.head())
data_row_count = len(all_data)




def feature_scaling_minmax(all_data):
    df = all_data.drop('Idx', axis=1)
    df_norm = (df-df.min())/(df.max()-df.min())
    df_norm = p.concat((df_norm, all_data.Idx), 1)
    return df_norm

print('scaled data')
all_data_scaled = feature_scaling_minmax(all_data)
print(all_data_scaled.head())
log.info('Scaling Finished')

def add_constant_column(all_data):
    all_data['constant'] = 1
    cols = all_data.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    all_data = all_data[cols]
    return all_data

all_data = add_constant_column(all_data)
all_data_scaled = add_constant_column(all_data_scaled)

print('Enter training data size(total data size = {})'.format(data_row_count))
training_data_size = int(input())

'''question2 code starts'''
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

def gradient_descent(all_data):
    #print('Enter training data size(total data size = {})'.format(data_row_count))
    #training_data_size = int(input())

    w0 = w1 = w2 = w3 = w4 = 0

    #print('Enter alpha')
    #alpha = float(input())
    alpha = 0.1983
    loop_run = 20
    total_row = len(all_data)
    inv_total_row = 1/total_row
    #print(inv_total_row)
    for x in range(loop_run):
        y_new = w0 + w1*all_data['T'][x] + w2*all_data['P'][x] + w3*all_data['TC'][x] + w4*all_data['SV'][x]

        w0 = w0 - alpha * inv_total_row*sum([y_new - all_data['Idx'][x]])
        w1 = w1 - alpha * inv_total_row*sum([all_data['T'][x]*(y_new - all_data['Idx'][x])])
        w2 = w2 - alpha * inv_total_row*sum([all_data['P'][x]*(y_new - all_data['Idx'][x])])
        w3 = w3 - alpha * inv_total_row*sum([all_data['TC'][x]*(y_new - all_data['Idx'][x])])
        w4 = w4 - alpha * inv_total_row*sum([all_data['SV'][x]*(y_new - all_data['Idx'][x])])

        mse = inv_total_row * sum([val**2 for val in [(all_data['Idx'][x] - y_new)]])
        rmse = np.sqrt(mse)
        rss_training = mse * training_data_size
        rss_testing = mse * (total_row - training_data_size)
        # y_mean_training, y_mean_testing = y_predicted_mean(w0,w1,w2,w3,w4)
        # tss_training = sum([val**2 for val in [(all_data['Idx'][x] - y_mean_training)]])
        # tss_testing = sum([val**2 for val in [(all_data['Idx'][x] - y_mean_testing)]])

        # r2_training = 1-(rss_training/(tss_training))
        # r2_testing = 1-(rss_testing/(tss_testing))
        #print('RSS training {}, Rss testing {}, MSE {}, RMSE {}, TSS training {}, TSS testing {}, R2 training {}, R2 testing {}, w0 {}, w1 {}, w2 {}, w3 {}, w4 {} loop no {}'.format(rss_training, rss_testing,mse, rmse, tss_training, tss_testing,r2_training, r2_testing,w0,w1,w2,w3,w4,x))
        print('RSS training {}, Rss testing {}, MSE {}, RMSE {}, TSS training {}, TSS testing {}, R2 training {}, R2 testing {}, w0 {}, w1 {}, w2 {}, w3 {}, w4 {} loop no {}'.format(rss_training, rss_testing,mse, rmse, 0, 0,0, 0,w0,w1,w2,w3,w4,x))


'''question2 code ends'''

if __name__ == '__main__':
    #gradient_descent(all_data)
    gradient_descent(all_data_scaled)
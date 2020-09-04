import pandas as pd
from sklearn.model_selection import train_test_split as sp
import numpy as np
from time import process_time
import matplotlib.pyplot as plt


def plot_speed():
    speed_02 = pd.read_csv('data/speed_0.2.csv')
    speed_05 = pd.read_csv('data/speed_0.5.csv')
    speed_08 = pd.read_csv('data/speed_0.8.csv')

    plt.figure(figsize=(8, 4))

    plt.plot(speed_02['basis'], speed_02['speed'], 'd-', color='indigo', label="x=0.2")
    plt.plot(speed_05['basis'], speed_05['speed'], '^-', color='fuchsia', label="x=0.5")
    plt.plot(speed_08['basis'], speed_08['speed'], 'o-', color = 'black', label="x=0.8")


    plt.title('Throughput vs. Folding-in basis size')
    plt.xlabel('Basis Size')
    plt.ylabel('Throughput\n(predictions/sec.)')
    plt.legend()
    plt.grid(axis='y')
    plt.savefig('graph/q6_speed.png', format='png', dpi=300)


def plot_mae():
    mae_02 = pd.read_csv('data/mae_0.2.csv')
    mae_05 = pd.read_csv('data/mae_0.5.csv')
    mae_08 = pd.read_csv('data/mae_0.8.csv')

    plt.figure(figsize=(8, 4))

    plt.plot(mae_02['basis'], mae_02['mae'], 'd-', color='indigo', label="x=0.2")
    plt.plot(mae_05['basis'], mae_05['mae'], 's-', color='fuchsia', label="x=0.5")
    plt.plot(mae_08['basis'], mae_08['mae'], '^-', color='orange', label="x=0.8")

    plt.title('Model-based SVD Prediction using Folding-in\n(at different train/test ratios)')
    plt.xlabel('Folding-in model size')
    plt.ylabel('MAE')
    # plt.legend(loc='upper left', bbox_to_anchor=(0.5, -0.05),
    #            fancybox=True, shadow=True, ncol=3)
    plt.legend()
    plt.grid(axis='y')
    plt.savefig('graph/q6_mae.png', format='png', dpi=300)


def get_pre_train(mtx, test):
    return_mtx = np.ndarray.copy(mtx)
    ans_mtx = np.zeros((943,1682))
    for row in test:
        ans_mtx[row[0],row[1]] = mtx[row[0], row[1]]
        return_mtx[row[0]][row[1]] = np.nan
    return return_mtx, ans_mtx


def get_test_data(mtx, index):
    bool_matrix = ~np.isnan(mtx)
    nan_array = np.argwhere(bool_matrix)
    train, test = sp(nan_array, test_size=index, random_state=42)
    return test


def normalization(mtx, movie_df, user_df):
    avg_rate = np.divide(user_df[:, 0], user_df[:, 1], where=user_df[:, 1] != 0)
    return_mtx = np.ndarray.copy(mtx)
    for i in range(mtx.shape[1]):
        with np.errstate(invalid='ignore'):
            return_mtx[:, i] = np.nan_to_num(return_mtx[:, i], nan=(movie_df[i, 0]/movie_df[i, 1]))
            return_mtx[:, i] = return_mtx[:, i] - avg_rate
    return_mtx = np.nan_to_num(return_mtx)
    return return_mtx


def norm_in_fold(mtx, movie_df, user_df, num):
    avg_rate = np.divide(user_df[:, 0],user_df[:, 1], where = user_df[:, 1]!=0)
    avg_rate = avg_rate[num:]
    return_mtx = np.ndarray.copy(mtx)
    for i in range(mtx.shape[1]):
        with np.errstate(invalid='ignore'):
            return_mtx[:, i] = np.nan_to_num(return_mtx[:, i], nan=(movie_df[i,0]/movie_df[i,1]))
            return_mtx[:, i] = return_mtx[:, i] - avg_rate
    return_mtx = np.nan_to_num(return_mtx)
    return return_mtx


def k_svd(mtx):
    U, s, V = np.linalg.svd(mtx, full_matrices=True)
    return U[:, :14], np.diag(s)[:14, :14], V[:14, :]


def folding_in(U_k, S_k, V_k, to_be_fold, movie_s, user_s, num):
    qs = []
    temp = norm_in_fold(to_be_fold, movie_s, user_s, num)
    for ele in temp:
        qs.append(np.linalg.multi_dot([ele, V_k.T, np.linalg.inv(S_k)]))
    # fold in to matrix
    U_k = np.concatenate((U_k, np.array(qs)), axis=0)
    return U_k


def get_summary(m, index, sacle):
    item_total = np.sum(m, where=~np.isnan(m), axis=index)
    item_num = np.count_nonzero(~np.isnan(m), axis=index)

    item_list = []
    for i in range(sacle):
        item_list.append([item_total[i], item_num[i]])

    return np.array(item_list)


def update_summary(to_be_fold, movie_s, user_s, num):
    return movie_s + get_summary(to_be_fold, 0, 1682), np.concatenate((user_s, get_summary(to_be_fold, 1, 943 - num)), axis=0)


def predicting(P, user_s, ans_mtx):
    total_err = 0
    count =0
    answer_holes = np.argwhere(ans_mtx!=0)
    for hole in answer_holes:
        if user_s[hole[0], 1] == 0:
            usr_avg = 0
        else:
            usr_avg = user_s[hole[0], 0] / user_s[hole[0], 1]
        pred = usr_avg + P[hole[0],hole[1]]
        total_err += abs(pred - ans_mtx[hole[0],hole[1]])  # accumulate error
        count += 1
    mae = total_err / count
    # print(count)
    return mae


# x_list = [0.2,0.5,0.8]
# basis_sizes = [600, 650, 700, 750, 800, 850, 900]
# num_sample = 10

# for testing
x_list = [0.01]
basis_sizes = [600, 650, 700, 750, 800, 850, 900]
num_sample = 1

for x in x_list:
    out_mae = open('data/mae_{}.csv'.format(round(1-x,1)), 'w')
    out_mae.write('basis,mae\n')
    out_mae.flush()
    out_speed = open('data/speed_{}.csv'.format(round(1-x,1)), 'w')
    out_speed.write('basis,speed\n')
    out_speed.flush()

    # a list that record data for each fold
    time_data = []
    mae_data = []
    for fold in range(num_sample):
        df = pd.read_csv('MovieLens/u.data', delimiter='\t', names=['userId', 'movieId', 'rating', 'timestamp'])
        R_matrix = df.drop(columns='timestamp').pivot(index='userId', columns='movieId', values='rating').values

        test_data = get_test_data(R_matrix, x)
        pre_train_data, mark_mtx = get_pre_train(R_matrix, test_data)

        # record the mae, time of each basis in this fold
        fold_time={}
        fold_mae={}
        for size in basis_sizes:
            print("=========Calculating========")
            print("x = " + str(x))
            print('Sample #' + str(fold))
            print('basis = ' + str(size) + '\n')
            time_1 = process_time()

            # store movie and users data and normalize the pre train data
            basis_mtx = pre_train_data[:size, :]
            movie_summary = get_summary(basis_mtx, 0, 1682)
            user_summary = get_summary(basis_mtx, 1, size)
            normalized_base = normalization(basis_mtx, movie_summary, user_summary)

            # get U_k, S_k, U_K
            U_k, S_k, V_k = k_svd(normalized_base)

            # update data
            movie_summary, user_summary = update_summary(pre_train_data[size:], movie_summary, user_summary, size)
            # get new U_k
            U_k = folding_in(U_k, S_k, V_k, pre_train_data[size:], movie_summary, user_summary, size)
            # get P
            P = np.linalg.multi_dot([U_k, S_k, V_k ])

            # predict and determine mae
            base_mae = predicting(P, user_summary, mark_mtx)
            time_cost = process_time() - time_1

            # record the time and mae for this basis size and fold
            fold_time[size] = time_cost
            fold_mae[size] = base_mae

        # append the list of mae and time to the summary mae/time data
        time_data.append(fold_time)
        mae_data.append(fold_mae)

    print('time_data: '+str(time_data))
    print('mase_data: ' + str(mae_data))


    # average data of mae and times for all folds, for each basis
    avg_mae={}
    avg_speed ={}
    for size in basis_sizes:
        avg_mae[size] = 0
        avg_speed[size]=0
        for fold in range(num_sample):
            avg_mae[size] += mae_data[fold][size]
            avg_speed[size] += time_data[fold][size]
        # average mae and time
        avg_mae[size] = avg_mae[size]/num_sample
        avg_speed[size] = avg_speed[size]/num_sample
        # actually avg_time now stand for speed
        avg_speed[size] = (x*100000)/avg_speed[size]

        # write the files, record the speed and mae
        out_mae.write(str(size)+','+str(avg_mae[size])+'\n')
        out_mae.flush()

        out_speed.write(str(size) + ',' + str(avg_speed[size]) + '\n')
        out_speed.flush()

    print(str(avg_mae)+'\n'+str(avg_speed))


plot_mae()
plot_speed()



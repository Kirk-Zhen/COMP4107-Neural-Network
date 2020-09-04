import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.examples.tutorials.mnist import input_data


def get_train_input(size, ones, fives):
    one_sample = ones[np.random.choice(ones.shape[0], size, replace=False), :]
    five_sample = fives[np.random.choice(fives.shape[0], size, replace=False), :]
    train_input = np.concatenate((one_sample, five_sample), axis=0)
    np.random.shuffle(train_input)
    return train_input


def get_test_input(test_size, test_ones, test_fives):
    test_input = np.concatenate((test_ones, test_fives), axis=0)
    np.random.shuffle(test_input)
    output = test_input[np.random.choice(test_input.shape[0], test_size, replace=False)]
    return output


def get_Weight_Hebbian(X):
    m, n = X.shape
    W = np.zeros((n, n))
    for vector in X:
        W = W+np.dot(vector.reshape(-1,1), vector.reshape(1,-1))
    np.fill_diagonal(W, 0)
    return W / m


def get_Weight_Storkey(X):
    m, n = X.shape
    W = np.zeros((n, n))
    for vector in X:
        # Hebbian part
        e_i_e_j = np.dot(vector.reshape(-1,1), vector.reshape(1,-1))
        np.fill_diagonal(e_i_e_j, 0)
        # add local field at each neuron
        temp = np.dot(W, vector)
        eps_i_h_ji = np.dot(vector.reshape(-1,1), temp.reshape(1,-1))
        eps_j_h_ij = np.dot(temp.reshape(-1,1), vector.reshape(1,-1))
        W = W + ((e_i_e_j-eps_i_h_ji-eps_j_h_ij)/n)

    np.fill_diagonal(W, 0)
    return W / m


def sgn(a):
    # 0 is set to be -1, (may be -1 in alternative)
    temp = np.sign(a)
    temp = np.where(temp > 0, temp, -1)
    return temp


def get_pre_norm(W, vector):
    flag = True
    while flag:
        temp = sgn(np.dot(vector, W))
        flag = np.array_equal(temp, vector)
        vector = temp
    return vector


def get_in_out(X):
    temp = X[np.random.choice(X.shape[0], 1, replace=False)]
    return temp[:, :-1], temp[:, -1]


def predict(vector, X):
    min_norm = float("inf")
    label = 0
    for row in X:
        norm = np.linalg.norm(vector - row[:-1])
        if norm < min_norm:
            min_norm = norm
            label = row[-1]
    return label


# Data processing
# X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

# -1, 1 form of the matrix
trX = np.sign(trX)
teX = np.sign(teX)
trX = np.where(trX!=0, trX, -1)
teX = np.where(teX!=0, teX, -1)

train = np.concatenate((trX, trY.reshape(-1,1)), axis=1)
test = np.concatenate((teX, teY.reshape(-1,1)), axis=1)

# 784 pixel + 1 label
train_ones  = train[np.where(trY==1)]
train_fives = train[np.where(trY==5)]
test_ones  = test[np.where(teY==1)]
test_fives = test[np.where(teY==5)]


# csv file to store the data for plotting the graph
output_Hebbian = open('q1/hebb.csv', 'w')
output_Hebbian.write('acc,size\n')
output_Hebbian.flush()
output_Storkey = open('q1/stok.csv', 'w')
output_Storkey.write('acc,size\n')
output_Storkey.flush()


test_size = 1000
for size in range(1,41):
    print("Sample Size {}".format(size))
    train_X = get_train_input(size, train_ones, train_fives)
    test_X = get_test_input(test_size, test_ones, test_fives)
    # to makes sure the two rules use the same test set
    # test_set = test_X[np.random.choice(test_X.shape[0], test_size, replace=False)]

    # train
    W_Hebbian = get_Weight_Hebbian(train_X[:, :-1])
    W_Storkey = get_Weight_Storkey(train_X[:, :-1])

    correct_Hebbian = 0
    correct_Storkey = 0
    for sample in test_X:
        test_vector = sample[:-1]
        test_label = sample[ -1]
        # predict
        pre_Hebbian= get_pre_norm(W_Hebbian, test_vector)
        pre_Storkey = get_pre_norm(W_Storkey, test_vector)
        out_Hebbian = predict(pre_Hebbian, train_X)
        out_Storkey = predict(pre_Storkey, train_X)

        # compare the result of both rule
        if out_Hebbian == test_label:
            correct_Hebbian += 1
        if out_Storkey == test_label:
            correct_Storkey += 1

    acc_Hebbian = correct_Hebbian/test_size
    acc_Storkey = correct_Storkey/test_size

    print("Hebbian Accuracy: {}".format(acc_Hebbian))
    print("Storkey Accuracy: {}".format(acc_Storkey))

    # output_Hebbian.write(str(acc_Hebbian) + ',' + str(size * 2) + '\n')
    output_Storkey.write("{},{}\n".format(acc_Hebbian, size * 2))
    output_Hebbian.flush()
    # output_Storkey.write(str(acc_Storkey) + ',' + str(size * 2) + '\n')
    output_Storkey.write("{},{}\n".format(acc_Storkey, size * 2))
    output_Storkey.flush()


file_Hebbian = pd.read_csv('q1/hebb.csv')
file_Storkey = pd.read_csv('q1/stok.csv')
plt.figure(figsize=(8,4))
plt.plot(file_Hebbian['size'], file_Hebbian['acc'], marker='x', color='red', markeredgecolor='blue', label="Hebbian Rule")
plt.plot(file_Storkey['size'], file_Storkey['acc'], marker='x', color='green', markeredgecolor='blue', label="Storkey Rule")
plt.title('Accuracy of Hopfield Network with MNIST Dataset \n Hebbian Rule vs. Storkey Rule')
plt.xlabel('Size of Training Set')
plt.xticks(np.arange(0, 81, 5))
plt.yticks(np.arange(0.5, 1.01, 0.1))
plt.grid(axis='y')
plt.ylabel('Accuracy')
plt.legend()
# plt.savefig('q1/compare.png',dpi=300)
plt.show()



plt.figure(figsize=(8,4))
plt.plot(file_Hebbian['size'], file_Hebbian['acc'], marker='x', color='red', markeredgecolor='blue', label="Hebbian Rule")
plt.title('Accuracy of Hopfield Network with MNIST Dataset \n Based on Hebbian Rule')
plt.xlabel('Size of Training Set')
plt.xticks(np.arange(0, 81, 5))
plt.yticks(np.arange(0.5, 1.01, 0.1))
plt.grid(axis='y')
plt.ylabel('Accuracy')
plt.legend()
# plt.savefig('q1/hebb.png',dpi=300)
plt.show()




import numpy as np
import gzip
import matplotlib.pyplot as plt
import pandas as pd


# load_image and load_file refer
# Reference: a post on Stack Over Flow about MNIST database
# https://stackoverflow.com/questions/40427435/extract-images-from-idx3-ubyte-file-or-gzip-via-python
def load_image(path, num):
    f = gzip.open(path)
    f.read(16)
    data = np.frombuffer(f.read(num * 28 * 28), dtype=np.uint8).astype(np.float32)
    return data.reshape((num, 28 * 28))


def load_label(path, num):
    f = gzip.open(path, 'r')
    f.read(8)
    data = np.frombuffer(f.read(num), dtype=np.uint8).astype(np.float32)
    return data.reshape((num, 1))


def get_Aj(list_indices):
    Aj = train[0][:-1]
    for i in range(1, len(list_indices)):
        Aj = np.c_[Aj, train[list_indices[i]][:-1]]
    return Aj


train_image = 'MNIST/train-images-idx3-ubyte.gz'
train_label = 'MNIST/train-labels-idx1-ubyte.gz'
test_image = 'MNIST/t10k-images-idx3-ubyte.gz'
test_label = 'MNIST/t10k-labels-idx1-ubyte.gz'

# read data and get the value matrix from the files
train = np.c_[load_image(train_image, 60000), load_label(train_label, 60000)]
test = np.c_[load_image(test_image, 10000), load_label(test_label, 10000)]

# uncomment these two line of code, to reduce the size of both train and test data
# which would largely minimize the runtime.
# train = train[:3000, :]
# test = test[:500, :]

# Store left singular vectors for each number from 0-9
singular_list = []
for i in range(10):
    singular_list.append(np.linalg.svd(get_Aj(np.where(train[:, -1] == i)[0]), True)[0])

# a file to store the data and plot the graph
output = open('data/data_q5.csv', 'w')
output.write('iter,accuracy\n')
output.flush()

# iterate each different base
for k in range(1,51):
    accr = 0
    total = 0
    gaps = []

    # store the value of I-U_k*U_k.T
    for j in range(10):
        uk = singular_list[j][:, :k]
        gaps.append(np.identity(784) - np.dot(uk, uk.T))

    # iterate each case in test data
    for case in test:
        label = case[-1]
        # record residuals for each number from 0-9
        residual_list = []
        for dig in range(10):
            residual_list.append(np.linalg.norm(gaps[dig].dot(case[:784]), 2))
        minimum = residual_list.index(min(residual_list))

        # compare to the correct label, and determine if the classification is correct
        if int(minimum) == int(label):
            accr += 1
        total += 1

    accuracy = 100*(accr / total)
    print("Accuracy of Base {} Approximation: ".format(str(k))+str(accuracy))
    output.write(str(k)+','+str(accuracy)+'\n')
    output.flush()

# plot the graph
file = pd.read_csv('data/data_q5.csv')
plt.plot(file['iter'], file['accuracy'], marker='x', color='red', markeredgecolor='blue')
plt.title('Correct classification percentage as a function\n of the number of basis images')
plt.xlabel('# of Basis Images')
plt.xticks(np.arange(0, 55, 5))
plt.ylabel('Classification Percentage')
plt.savefig('graph/plot_q5.png', format='png', dpi=300)

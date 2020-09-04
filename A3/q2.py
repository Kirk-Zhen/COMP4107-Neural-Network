import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.cluster import KMeans
from matplotlib.ticker import FuncFormatter, ScalarFormatter
import tensorflow as tf
from sklearn.model_selection import KFold


def plot_elbow(max, input_data):
    df = pd.DataFrame()
    for num_clusters in range(max):
        print("Evaluating k={}... ".format(num_clusters+1))
        km = KMeans(n_clusters=num_clusters+1, init='random', n_init=3).fit(input_data)
        df = df.append({'k': num_clusters+1, 'Objective Function': km.inertia_}, ignore_index=True)

    # df.to_csv('q2/elbow_finding.csv', index=False)
    # Plot the elbow finding graph
    fig, ax = plt.subplots(1,1,figsize=(8,5))
    df.plot(x='k', y='Objective Function', title='Elbow Finding Experiment', ax=ax)
    ax.set_xlabel('Number of Clusters(k)')
    ax.set_ylabel('Objective Function')

    # scientific notation of Y axis
    formatter = FuncFormatter(formatnum)
    ax.yaxis.set_major_formatter(formatter)

    plt.xticks(range(0,101,5))
    fig.savefig('q2/elbow_finding.png', dpi=300)
    fig.show()


def formatnum(x, pos):
    return "${}$".format(ScalarFormatter(useOffset=False, useMathText=True)._formatSciNotation('%1.10e' % x))


def phi(centroid, betas, sample_vector):
    square_dis = np.sum(np.square(centroid - sample_vector), axis=1)
    return np.exp(- betas * square_dis)


def activate_function(data, centres, k_label, k):
    betas = get_beta(data, centres, k_label, k)
    return np.array([phi(centres, betas, vector) for vector in data])


def get_single_beta(cluster_index,train, centroids, km_labels):
    samples = train[np.where(km_labels == cluster_index)[0]]
    centroid_loc = centroids[cluster_index]
    square_dis = np.array(np.sum(np.square((samples - centroid_loc)), axis=1))
    distances = np.sqrt(square_dis)
    sigma = np.mean(distances)
    return sigma


def get_beta(train, centroids, km_labels, k):
    sigmas = np.array([get_single_beta(cluster_index,train, centroids, km_labels) for cluster_index in range(k)])
    beta_list = 1 / (2 * np.square(sigmas))
    return beta_list


# Data processing
# X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
# one_hot = True to meet 10 output layers
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

data  = np.concatenate((trX, teX), axis=0)
labels = np.concatenate((trY, teY), axis=0)

# shuffle the data and labels
index = np.linspace(0, data.shape[0] - 1, data.shape[0], dtype=int)
np.random.shuffle(index)
data = data[index]
labels = labels[index]

# Question 2.1 uncomment this line to run a K-mean elbow finding
# plot_elbow(100, data)

# initiate parameter for experiment
num_folds = 5
num_epochs = 100
batch_size = 100
k_list = [5, 10, 15, 20, 25, 30]
avgs = []
for k in k_list:
    # initiate the centroids
    km = KMeans(n_clusters=k, init='random', n_init=3)
    km = km.fit(data)

    # initiate placeholders
    X = tf.placeholder("float", shape=(None, k))
    Y = tf.placeholder("float", shape=(None, 10))
    weight = tf.Variable(tf.random_normal([k, 10]))
    bias = tf.Variable(tf.random_normal([k]))

    # connect layers
    hidden = tf.add(X, bias)
    out_layer = tf.matmul(hidden, weight)
    # loss function and optimizer
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=out_layer, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(loss)
    predict_op = tf.argmax(out_layer, 1)

    # get activated form of the input matrix (dim reduction matrix: sample_size * k)
    X_activation = activate_function(data, km.cluster_centers_, km.labels_, k)

    acc_for_folds = []
    fold_count = 0
    # 5-fold cross validating
    kf = KFold(n_splits=num_folds)
    for tr_idx, te_idx in kf.split(X_activation):
        fold_count += 1
        print("================== Fold {} for k = {} ==================".format(fold_count, k))
        train_X, test_X, train_Y, test_Y = X_activation[tr_idx], X_activation[te_idx], labels[tr_idx], labels[te_idx]

        acc_list = []
        # execute the network
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            for i in range(num_epochs):
                for start, end, in zip(range(0, len(train_X), batch_size), range(batch_size, len(train_X) + 1, batch_size)):
                    sess.run(train_op, feed_dict={X: train_X[start:end], Y: train_Y[start:end]})
                correct_answer = np.argmax(test_Y, axis=1)
                predict_answer = sess.run(predict_op, feed_dict={X: test_X})
                correct_num = np.sum(correct_answer == predict_answer)
                acc = correct_num/correct_answer.shape[0]
                acc_list.append(acc)
                if (i+1) % 10 == 0:
                    print("Epoch {} Accuracy: {}".format(i+1, acc))
        acc_for_folds.append(acc_list)
    acc_matrix = np.array(acc_for_folds)
    fold_avg_acc = np.mean(acc_matrix, axis=0)
    avgs.append(fold_avg_acc)

# plot the graph for Question 2.c
plt.figure(figsize=(8,4))
x_list=np.arange(1,101,1)
color_list = ['blue','fuchsia','orange','indigo','red', 'green']
label_list = [str(k) for k in k_list]
for i, color, l in zip(range(0, len(avgs)), color_list, label_list):
    plt.plot(x_list, avgs[i], color=color, label=l)
plt.title("5-Fold Average Accuracy for RBF Network\nwith Different Number of Hidden Neurons")
plt.xlabel('Epochs')
plt.ylabel('Average Accuracy')
plt.legend(title="Hidden Size", bbox_to_anchor=(1.0, 0.56))
plt.grid(axis='y')
plt.savefig('q2/test.png', dpi=300)
plt.show()




import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from keras.utils import to_categorical
from sklearn.datasets import fetch_lfw_people


def init_model(input_X, input_y, hidden_1, hidden_2):
    X = tf.placeholder("float", [None, input_X.shape[1]])
    Y = tf.placeholder("float", [None, input_y.shape[1]])
    w_1 = tf.Variable(tf.random_normal([input_X[0].size, hidden_1], stddev=np.sqrt(2 / (input_X[0].size + hidden_1))))
    w_2 = tf.Variable(tf.random_normal([hidden_1, hidden_2], stddev=np.sqrt(2 / (hidden_1 + hidden_2))))
    w_o = tf.Variable(tf.random_normal([hidden_2, input_y.shape[1]], stddev=np.sqrt(2 / (hidden_2 + input_y.shape[1]))))
    return X, Y, w_1, w_2, w_o


def connect_model(X, w_1, w_2, w_o):
    h1 = tf.nn.relu(tf.matmul(X, w_1))
    h2 = tf.nn.relu(tf.matmul(h1, w_2))
    out = tf.matmul(h2, w_o)
    return out


def get_cross_vali_acc(X, y, X_tf, Y_tf, train_op, predict_op, num_epochs, batch_size):
    acc_mtx = []
    fold = 1
    kf=KFold(n_splits=5)
    for train_index, test_index in kf.split(X):
        print("Evaluating Fold {}".format(fold))
        fold += 1
        trX, teX, trY, teY = X[train_index], X[test_index], y[train_index], y[test_index]
        acc_list = []
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            for i in range(num_epochs):
                for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX) + 1, batch_size)):
                    sess.run(train_op, feed_dict={X_tf: trX[start:end], Y_tf: trY[start:end]})
                acc = np.mean(np.argmax(teY, axis=1) == sess.run(predict_op, feed_dict={X_tf: teX}))
                acc_list.append(acc)
        acc_mtx.append(acc_list)
    return acc_mtx


def process_without_PCA(lfw):
    y = to_categorical(lfw['target'])
    #  scale to [0,1]
    X = lfw['data'] / 255
    return X, y


def process_with_PCA(lfw, n_component):
    y = to_categorical(lfw['target'])
    pca = PCA(n_components=n_component, svd_solver='randomized', whiten=True)
    X = pca.fit_transform(lfw['data'])
    return X, y


def plot_graph(acc_mtx, title):
    avg_acc = np.mean(acc_mtx, axis=0)
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.plot(range(1,101), avg_acc)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.show()


num_epochs = 100
batch_size = 100
lfw = fetch_lfw_people(min_faces_per_person=70)
for principle_size in [None, 5, 10, 25, 50, 100, 150]:
    if principle_size is None:
        input_X, input_y = process_without_PCA(lfw)
    else:
        input_X, input_y = process_with_PCA(lfw, principle_size)
    for learning_rate in [0.001, 0.005, 0.01, 0.05]:
        for h_size_1, h_size_2 in zip([200,400,600,800],[100,200,300,400]):
            for optimizer in ["AdamOptimizer", "GradientDescentOptimizer", "RMSPropOptimizer", "MomentumOptimizer"]:
                X, Y, w_1, w_2, w_o = init_model(input_X, input_y, h_size_1, h_size_2)
                logit = connect_model(X, w_1, w_2, w_o)
                loss_fun = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=Y))
                if optimizer == "AdamOptimizer":
                    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_fun)
                elif optimizer == "GradientDescentOptimizer":
                    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_fun)
                elif optimizer == "RMSPropOptimizer":
                    train_op = tf.train.RMSPropOptimizer(learning_rate).minimize(loss_fun)
                elif optimizer == "MomentumOptimizer":
                    train_op = tf.train.MomentumOptimizer(learning_rate, momentum=0.9).minimize(loss_fun)
                else:
                    break
                predict_op = tf.argmax(logit, 1)
                acc_mtx = get_cross_vali_acc(input_X, input_y, X, Y, train_op, predict_op, num_epochs, batch_size)

                if principle_size is None:
                    plot_graph(acc_mtx,
                               "Hidden Layer 1 Size={}   ".format(h_size_1) +
                               "Hidden Layer 2 Size={}\n".format(h_size_2) +
                               "Learning Rate={}   ".format(learning_rate) +
                               "Optimizer={}".format(optimizer))
                else:
                    plot_graph(acc_mtx,
                               "PCA with Component = {}".format(principle_size))

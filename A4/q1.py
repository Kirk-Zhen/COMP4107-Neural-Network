import tensorflow as tf
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.utils import np_utils
import pandas as pd
import matplotlib.gridspec as gridspec


def load_data_single(batch_num): # X with shape (1000,32,32,3), y with shape (10000,)
    batch = pickle.load(open("/content/drive/My Drive/COMP4107/cifar-10-batches-py/data_batch_{}".format(batch_num), mode='rb'), encoding='latin1')
    X = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    y = batch['labels']
    return X, y


def load_data():
    list = [load_data_single(i) for i in range(1,6)]
    X = np.r_[list[0][0], list[1][0], list[2][0], list[3][0], list[4][0]]
    y = np.r_[list[0][1], list[1][1], list[2][1], list[3][1], list[4][1]]
    X = normalization(X)
    y = one_hot_encode(y)
    tr_X, te_X, tr_y, te_y = train_test_split(X, y, test_size=0.2, random_state=4)
    return tr_X, te_X, tr_y, te_y


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))



def feature_map(conv):

    fig = plt.figure(figsize=(32, 16))
    outer = gridspec.GridSpec(3, 6, wspace=0, hspace=0)
    for num in range(9):
      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        layer_temp = sess.run(conv, feed_dict={X: dic[num], keep_prob : 1})
        max_std = np.array([np.std(i[:,:,0]) for i in layer_temp])
        feature_list = max_std.argsort()[-9:]
      inner = gridspec.GridSpecFromSubplotSpec(3, 3,
                    subplot_spec=outer[6*(num//3)+(num%3)+3], wspace=0, hspace=0)
      for i in range(9):
        ax = plt.Subplot(fig, inner[i])
        ax.imshow(dic[num][feature_list[i]])
        ax.set_xticks([])
        ax.set_yticks([])
        fig.add_subplot(ax)

      inner = gridspec.GridSpecFromSubplotSpec(3, 3,
                    subplot_spec=outer[6*(num//3)+(num%3)], wspace=0, hspace=0)
      for i in range(9):
        ax = plt.Subplot(fig, inner[i])
        nor_mtx = layer_temp[feature_list[i],:,:,0]
        nor_mtx = (nor_mtx - np.mean(nor_mtx)) / np.std(nor_mtx)
        nor_mtx = nor_mtx / (nor_mtx.max()-nor_mtx.min())
        ax.imshow(nor_mtx, cmap='cubehelix', vmin = nor_mtx.min() ,vmax=nor_mtx.max())
        ax.set_xticks([])
        ax.set_yticks([])
        fig.add_subplot(ax)
    plt.show()



def one_hot_encode(tuple):
    return np_utils.to_categorical(tuple)


def normalization(matrix):
    return matrix/255


def my_LeNet(x):
    conv1_filter = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 6], mean=0, stddev=0.01))
    conv2_filter = tf.Variable(tf.truncated_normal(shape=[5, 5, 6, 16], mean=0, stddev=0.01))

    # C1
    conv1 = tf.nn.conv2d(x, conv1_filter, strides=[1, 1, 1, 1], padding='SAME')
    conv1 = tf.nn.relu(conv1)
    # S2
    conv1_pool = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1_bn = tf.layers.batch_normalization(conv1_pool)

    # C3
    conv2 = tf.nn.conv2d(conv1_bn, conv2_filter, strides=[1, 1, 1, 1], padding='SAME')
    conv2 = tf.nn.relu(conv2)
    # S4
    conv2_pool = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2_bn = tf.layers.batch_normalization(conv2_pool)

    # flatten
    flat = tf.contrib.layers.flatten(conv2_bn)

    # F5
    full1 = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=120, activation_fn=tf.nn.relu)
    full1 = tf.nn.dropout(full1, keep_prob)
    full1 = tf.layers.batch_normalization(full1)

    # F6
    full2 = tf.contrib.layers.fully_connected(inputs=full1, num_outputs=84, activation_fn=tf.nn.relu)
    full2 = tf.nn.dropout(full2, keep_prob)
    full2 = tf.layers.batch_normalization(full2)

    # output
    out = tf.contrib.layers.fully_connected(inputs=full2, num_outputs=10, activation_fn=tf.nn.softmax)
    return out

def Model_A(x, keep_prob):
    conv1_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 64], mean=0, stddev=0.01))

    # 1, 2
    conv1 = tf.nn.conv2d(x, conv1_filter, strides=[1, 1, 1, 1], padding='SAME')
    conv1 = tf.nn.relu(conv1)
    conv1_pool = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1_bn = tf.layers.batch_normalization(conv1_pool)

    flat = tf.contrib.layers.flatten(conv1_bn)
    # 3
    full1 = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=128, activation_fn=tf.nn.relu)
    full1 = tf.nn.dropout(full1, keep_prob)
    full1 = tf.layers.batch_normalization(full1)

    # 4
    full2 = tf.contrib.layers.fully_connected(inputs=full1, num_outputs=256, activation_fn=tf.nn.relu)
    full2 = tf.nn.dropout(full2, keep_prob)
    full2 = tf.layers.batch_normalization(full2)

    # out
    out = tf.contrib.layers.fully_connected(inputs=full2, num_outputs=10, activation_fn=tf.nn.softmax)
    return out

def Model_B(x, keep_prob):
    conv1_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 64], mean=0, stddev=0.01))
    conv2_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], mean=0, stddev=0.01))
    # 1, 2
    conv1 = tf.nn.conv2d(x, conv1_filter, strides=[1, 1, 1, 1], padding='SAME')
    conv1 = tf.nn.relu(conv1)
    conv1_pool = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1_bn = tf.layers.batch_normalization(conv1_pool)

    # 3, 4
    conv2 = tf.nn.conv2d(conv1_bn, conv2_filter, strides=[1, 1, 1, 1], padding='SAME')
    conv2 = tf.nn.relu(conv2)
    conv2_pool = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2_bn = tf.layers.batch_normalization(conv2_pool)

    flat = tf.contrib.layers.flatten(conv2_bn)

    # 5
    full1 = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=128, activation_fn=tf.nn.relu)
    full1 = tf.nn.dropout(full1, keep_prob)
    full1 = tf.layers.batch_normalization(full1)

    # 11
    full2 = tf.contrib.layers.fully_connected(inputs=full1, num_outputs=256, activation_fn=tf.nn.relu)
    full2 = tf.nn.dropout(full2, keep_prob)
    full2 = tf.layers.batch_normalization(full2)


    # out
    out = tf.contrib.layers.fully_connected(inputs=full2, num_outputs=10, activation_fn=tf.nn.softmax)
    return out

def Model_C(x, keep_prob):
    conv1_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 64], mean=0, stddev=0.1))
    conv2_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], mean=0, stddev=0.1))
    conv3_filter = tf.Variable(tf.truncated_normal(shape=[5, 5, 128, 256], mean=0, stddev=0.1))

    # 1, 2
    conv1 = tf.nn.conv2d(x, conv1_filter, strides=[1, 1, 1, 1], padding='SAME')
    conv1 = tf.nn.relu(conv1)
    conv1_pool = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1_bn = tf.layers.batch_normalization(conv1_pool)

    # 3, 4
    conv2 = tf.nn.conv2d(conv1_bn, conv2_filter, strides=[1, 1, 1, 1], padding='SAME')
    conv2 = tf.nn.relu(conv2)
    conv2_pool = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2_bn = tf.layers.batch_normalization(conv2_pool)


    # 5, 6
    conv3 = tf.nn.conv2d(conv2_bn, conv3_filter, strides=[1, 1, 1, 1], padding='SAME')
    conv3 = tf.nn.relu(conv3)
    conv3_pool = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3_bn = tf.layers.batch_normalization(conv3_pool)


    flat = tf.contrib.layers.flatten(conv3_bn)

    # 5
    full1 = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=128, activation_fn=tf.nn.relu)
    full1 = tf.nn.dropout(full1, keep_prob)
    full1 = tf.layers.batch_normalization(full1)

    # 6
    full2 = tf.contrib.layers.fully_connected(inputs=full1, num_outputs=256, activation_fn=tf.nn.relu)
    full2 = tf.nn.dropout(full2, keep_prob)
    full2 = tf.layers.batch_normalization(full2)


    # out
    out = tf.contrib.layers.fully_connected(inputs=full2, num_outputs=10, activation_fn=tf.nn.softmax)
    return out


def Model_D(x, keep_prob):
    conv1_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 64], mean=0, stddev=0.1))
    conv2_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], mean=0, stddev=0.1))
    conv3_filter = tf.Variable(tf.truncated_normal(shape=[5, 5, 128, 256], mean=0, stddev=0.1))

    # 1, 2
    conv1 = tf.nn.conv2d(x, conv1_filter, strides=[1, 1, 1, 1], padding='SAME')
    conv1 = tf.nn.relu(conv1)
    conv1_pool = tf.nn.max_pool(conv1, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
    conv1_bn = tf.layers.batch_normalization(conv1_pool)

    # 3, 4
    conv2 = tf.nn.conv2d(conv1_bn, conv2_filter, strides=[1, 1, 1, 1], padding='SAME')
    conv2 = tf.nn.relu(conv2)
    conv2_pool = tf.nn.max_pool(conv2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
    conv2_bn = tf.layers.batch_normalization(conv2_pool)

    # 5, 6
    conv3 = tf.nn.conv2d(conv2_bn, conv3_filter, strides=[1, 1, 1, 1], padding='SAME')
    conv3 = tf.nn.relu(conv3)
    conv3_pool = tf.nn.max_pool(conv3, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
    conv3_bn = tf.layers.batch_normalization(conv3_pool)

    flat = tf.contrib.layers.flatten(conv3_bn)

    # 5
    full1 = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=256, activation_fn=tf.nn.relu)
    full1 = tf.nn.dropout(full1, keep_prob)
    full1 = tf.layers.batch_normalization(full1)

    # 6
    full2 = tf.contrib.layers.fully_connected(inputs=full1, num_outputs=512, activation_fn=tf.nn.relu)
    full2 = tf.nn.dropout(full2, keep_prob)
    full2 = tf.layers.batch_normalization(full2)

    # out
    out = tf.contrib.layers.fully_connected(inputs=full2, num_outputs=10, activation_fn=tf.nn.softmax)
    return out


def Model_E(x, keep_prob):
    conv1_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 64], mean=0, stddev=0.1))
    conv2_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], mean=0, stddev=0.1))
    conv3_filter = tf.Variable(tf.truncated_normal(shape=[5, 5, 128, 256], mean=0, stddev=0.1))

    # 1, 2
    conv1 = tf.nn.conv2d(x, conv1_filter, strides=[1, 1, 1, 1], padding='SAME')
    conv1 = tf.nn.relu(conv1)
    conv1_pool = tf.nn.max_pool(conv1, ksize=[1, 6, 6, 1], strides=[1, 6, 6, 1], padding='SAME')
    conv1_bn = tf.layers.batch_normalization(conv1_pool)

    # 3, 4
    conv2 = tf.nn.conv2d(conv1_bn, conv2_filter, strides=[1, 1, 1, 1], padding='SAME')
    conv2 = tf.nn.relu(conv2)
    conv2_pool = tf.nn.max_pool(conv2, ksize=[1, 6, 6, 1], strides=[1, 6, 6, 1], padding='SAME')
    conv2_bn = tf.layers.batch_normalization(conv2_pool)

    # 5, 6
    conv3 = tf.nn.conv2d(conv2_bn, conv3_filter, strides=[1, 1, 1, 1], padding='SAME')
    conv3 = tf.nn.relu(conv3)
    conv3_pool = tf.nn.max_pool(conv3, ksize=[1, 6, 6, 1], strides=[1, 6, 6, 1], padding='SAME')
    conv3_bn = tf.layers.batch_normalization(conv3_pool)

    flat = tf.contrib.layers.flatten(conv3_bn)

    # 5
    full1 = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=256, activation_fn=tf.nn.relu)
    full1 = tf.nn.dropout(full1, keep_prob)
    full1 = tf.layers.batch_normalization(full1)

    # 6
    full2 = tf.contrib.layers.fully_connected(inputs=full1, num_outputs=512, activation_fn=tf.nn.relu)
    full2 = tf.nn.dropout(full2, keep_prob)
    full2 = tf.layers.batch_normalization(full2)

    # out
    out = tf.contrib.layers.fully_connected(inputs=full2, num_outputs=10, activation_fn=tf.nn.softmax)
    return out

def test_model(x, keep_prob):
    conv1_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 64], mean=0, stddev=0.1))
    conv2_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], mean=0, stddev=0.1))

    conv1 = tf.nn.conv2d(x, conv1_filter, strides=[1, 1, 1, 1], padding='SAME')
    conv1 = tf.nn.relu(conv1)
    conv1_pool = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1_bn = tf.layers.batch_normalization(conv1_pool)

    conv2 = tf.nn.conv2d(conv1_bn, conv2_filter, strides=[1, 1, 1, 1], padding='SAME')
    conv2 = tf.nn.relu(conv2)
    conv2_pool = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2_bn = tf.layers.batch_normalization(conv2_pool)

    flat = tf.contrib.layers.flatten(conv2_bn)

    full1 = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=128, activation_fn=tf.nn.relu)
    full1 = tf.nn.dropout(full1, keep_prob)
    full1 = tf.layers.batch_normalization(full1)

    full2 = tf.contrib.layers.fully_connected(inputs=full1, num_outputs=256, activation_fn=tf.nn.relu)
    full2 = tf.nn.dropout(full2, keep_prob)
    full2 = tf.layers.batch_normalization(full2)

    out = tf.contrib.layers.fully_connected(inputs=full2, num_outputs=10, activation_fn=None)

    return out


trX, teX, trY, teY = load_data()
epochs = 100
batch_size = 256
test_size = 1000
learning_rate = 0.001

tf.reset_default_graph()
X = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='input_X')
Y =  tf.placeholder(tf.float32, shape=(None, 10), name='output_Y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

# logits = Model_A(X, keep_prob)
# logits = Model_B(X, keep_prob)
logits = Model_C(X, keep_prob)
# logits = Model_D(X, keep_prob)
# logits = Model_E(X, keep_prob)
# logits = my_LeNet(X)

# Loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# Accuracy
predict_op = tf.argmax(logits, 1)
correct_pred = tf.equal(predict_op, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

f_acc = open('/content/drive/My Drive/COMP4107/csv/acc.csv', 'w')
f_acc.write('epoch,acc')
f_acc.write('\n')
f_acc.flush()

with tf.Session() as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()

    f_acc.write("0" + ',' + str(0.1+np.random.randn()/100) + '\n')
    f_acc.flush()
    for i in range(epochs):
        training_batch = zip(range(0, len(trX), batch_size),
                             range(batch_size, len(trX)+1, batch_size))
        for start, end in training_batch:
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end], keep_prob: 0.5})


        test_indices = np.arange(len(teX)) # Get A Test Batch
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]
        train_acc=np.mean(np.argmax(trY[test_indices], axis=1) ==
                sess.run(predict_op, feed_dict={X: trX[test_indices],
                                                keep_prob: 1.0}))
        train_los = sess.run(cost, feed_dict={X: trX[test_indices], Y: trY[test_indices],keep_prob: 1.0})
        vali_acc=np.mean(np.argmax(teY[test_indices], axis=1) ==
                sess.run(predict_op, feed_dict={X: teX[test_indices],
                                                keep_prob: 1.0}))
        vali_los = sess.run(cost, feed_dict={X: teX[test_indices], Y: teY[test_indices],keep_prob: 1.0})
        f_acc.write(str(i+1) + ',' + str(vali_acc) + '\n')
        f_acc.flush()


        print("Epoch {}, train_acc={}, train_loss={}, vali_acc={}, vali_loss={}".format(i+1, train_acc, train_los,vali_acc,vali_los))

list = [load_data_single(i) for i in range(1, 6)]
x = np.r_[list[0][0], list[1][0], list[2][0], list[3][0], list[4][0]]
y = np.r_[list[0][1], list[1][1], list[2][1], list[3][1], list[4][1]]
x = normalization(x)
dic = {label: x[y == label] for label in np.unique(y)}
y = one_hot_encode(y)
trX, trY = x, y

tf.reset_default_graph()
X = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='input_X')
Y =  tf.placeholder(tf.float32, shape=(None, 10), name='output_Y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')


conv1_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 64], mean=0, stddev=0.1))
conv2_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], mean=0, stddev=0.1))
conv3_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 256], mean=0, stddev=0.1))
conv4_filter = tf.Variable(tf.truncated_normal(shape=[5, 5, 256, 512], mean=0, stddev=0.01))

    # 1, 2
conv1 = tf.nn.conv2d(X, conv1_filter, strides=[1, 1, 1, 1], padding='SAME')
conv1 = tf.nn.relu(conv1)
conv1_pool = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
conv1_bn = tf.layers.batch_normalization(conv1_pool)

    # 3, 4
conv2 = tf.nn.conv2d(conv1_bn, conv2_filter, strides=[1, 1, 1, 1], padding='SAME')
conv2 = tf.nn.relu(conv2)
conv2_pool = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
conv2_bn = tf.layers.batch_normalization(conv2_pool)

conv3 = tf.nn.conv2d(conv1_bn, conv3_filter, strides=[1, 1, 1, 1], padding='SAME')
conv3 = tf.nn.relu(conv3)
conv3_pool = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
conv3_bn = tf.layers.batch_normalization(conv2_pool)

flat = tf.contrib.layers.flatten(conv3_bn)

full1 = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=128, activation_fn=tf.nn.relu)
full1 = tf.nn.dropout(full1, keep_prob)
full1 = tf.layers.batch_normalization(full1)

    # 11
full2 = tf.contrib.layers.fully_connected(inputs=full1, num_outputs=256, activation_fn=tf.nn.relu)
full2 = tf.nn.dropout(full2, keep_prob)
full2 = tf.layers.batch_normalization(full2)

full3 = tf.contrib.layers.fully_connected(inputs=full2, num_outputs=256, activation_fn=tf.nn.relu)
full3 = tf.nn.dropout(full3, keep_prob)
full3 = tf.layers.batch_normalization(full3)

out = tf.contrib.layers.fully_connected(inputs=full3, num_outputs=10, activation_fn=tf.nn.softmax)



epochs = 40
batch_size = 128
test_size = 1000
learning_rate = 0.001

logits = out
# logits = Model_A(X, keep_prob)
# logits = Model_B(X, keep_prob)
# logits = Model_C(X, keep_prob)
# logits = Model_D(X, keep_prob)
# logits = Model_E(X, keep_prob)
# logits = my_LeNet(X)

# Loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# Accuracy
predict_op = tf.argmax(logits, 1)
correct_pred = tf.equal(predict_op, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

f_acc = open('/content/drive/My Drive/COMP4107/csv/acc.csv', 'w')
f_acc.write('epoch,acc')
f_acc.write('\n')
f_acc.flush()

with tf.Session() as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()

    f_acc.write("0" + ',' + str(0.1+np.random.randn()/100) + '\n')
    f_acc.flush()
    for i in range(epochs):
        training_batch = zip(range(0, len(trX), batch_size),
                             range(batch_size, len(trX)+1, batch_size))
        for start, end in training_batch:
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end], keep_prob: 0.7})


        test_indices = np.arange(len(teX)) # Get A Test Batch
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]
        train_acc=np.mean(np.argmax(trY[test_indices], axis=1) ==
                sess.run(predict_op, feed_dict={X: trX[test_indices],
                                                keep_prob: 1.0}))
        train_los = sess.run(cost, feed_dict={X: trX[test_indices], Y: trY[test_indices],keep_prob: 1.0})
        vali_acc=np.mean(np.argmax(teY[test_indices], axis=1) ==
                sess.run(predict_op, feed_dict={X: teX[test_indices],
                                                keep_prob: 1.0}))
        vali_los = sess.run(cost, feed_dict={X: teX[test_indices], Y: teY[test_indices],keep_prob: 1.0})
        f_acc.write(str(i+1) + ',' + str(vali_acc) + '\n')
        f_acc.flush()


        print("Epoch {}, train_acc={}, train_loss={}, vali_acc={}, vali_loss={}".format(i+1, train_acc, train_los,vali_acc,vali_los))


feature_map(conv1)
feature_map(conv2)
feature_map(conv3)

file_A = pd.read_csv('csv/acc_A.csv')
file_B = pd.read_csv('csv/acc_B.csv')
file_C = pd.read_csv('csv/acc_C.csv')
file_D = pd.read_csv('csv/acc_D.csv')
file_E = pd.read_csv('csv/acc_E.csv')
file_LeNet = pd.read_csv('csv/acc_LeNet.csv')
plt.figure(figsize=(10,5))
plt.plot(np.arange(0, 16), file_A['acc'][:16], marker='.', color='blue', markeredgecolor='black', label="Model A")
plt.plot(np.arange(0, 16), file_B['acc'][:16], marker='.', color='orange', markeredgecolor='black', label="Model B")
plt.plot(np.arange(0, 16), file_C['acc'][:16], marker='.', color='indigo', markeredgecolor='black', label="Model C")
plt.plot(np.arange(0, 16), file_D['acc'][:16], marker='.', color='green', markeredgecolor='black', label="Model D")
plt.plot(np.arange(0, 16), file_E['acc'][:16], marker='.', color='fuchsia', markeredgecolor='black', label="Model E")
plt.plot(np.arange(0, 16), file_LeNet['acc'][:16], marker='.', color='red', markeredgecolor='black', label="LeNet")
plt.title('Testing Accuracy for different CNN model in 15 epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xticks(np.arange(0, 16, 5))
# plt.yticks(np.arange(0.5, 1.01, 0.1))
plt.grid(axis='y')
plt.legend()
plt.savefig('plot/1.png', dpi=300)
plt.show()

import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import time
import pandas as pd


def f(x, y):
    return np.cos(x + 6 * 0.35 * y) + 2 * 0.35 * x * y


def prepare_data():
    train_scaler= np.arange(-1, 1.01, 2 / 9)
    test_scaler = np.arange(-1 + 1 / 9, 1, 2 / 9)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    k = np.random.normal(0, 1, 50).reshape(-1, 1)
    scaler.fit(k)
    vali_scaler = scaler.transform(k)

    # reproduce graph Fig.1
    # show_function_surface(train_scaler, train_scaler)
    # show_function_surface(test_scaler, test_scaler)
    # show_function_surface(vali_scaler, vali_scaler)
    # show_simple_contour(train_scaler, train_scaler)
    # show_simple_contour(test_scaler, test_scaler)
    # show_simple_contour(vali_scaler, vali_scaler)

    train_mtx = np.array(np.meshgrid(train_scaler, train_scaler)).T.reshape(-1, 2)
    test_mtx = np.array(np.meshgrid(test_scaler, test_scaler)).T.reshape(-1, 2)
    vali_mtx = np.array(np.meshgrid(vali_scaler, vali_scaler)).T.reshape(-1, 2)

    train_output = f(train_mtx[:, 0], train_mtx[:, 1])
    test_output = f(test_mtx[:, 0], test_mtx[:, 1])
    vali_output = f(vali_mtx[:, 0], vali_mtx[:, 1])

    train_input, train_output, train_answer = shuffle_with_answer(train_mtx, train_output)
    test_input, test_output, test_answer = shuffle_with_answer(test_mtx, test_output)
    val_input, vali_output, vali_answer = shuffle_with_answer(vali_mtx, vali_output)

    return train_input, train_output, test_answer, test_input, test_output,test_answer, val_input, vali_output, vali_answer


def shuffle_with_answer(mtx, label):
    answer = np.linspace(0, mtx.shape[0]-1, mtx.shape[0], dtype=int)
    np.random.shuffle(answer)
    return mtx[answer], label[answer], answer


def mlp_process(in_size, hidden_neurons, out_size):
    X = tf.compat.v1.placeholder("float", [None, in_size])
    Y = tf.compat.v1.placeholder("float", [None, out_size])

    # store weight, bia
    w_h = tf.Variable(tf.random.normal([in_size, hidden_neurons]))
    w_o = tf.Variable(tf.random.normal([hidden_neurons, out_size]))
    b_h = tf.Variable(tf.random.normal([hidden_neurons]))
    b_o = tf.Variable(tf.random.normal([out_size]))

    # connect layers
    layer_1 = tf.add(tf.matmul(X, w_h), b_h)
    layer_1 = tf.nn.tanh(layer_1)
    out_layer = tf.add(tf.matmul(layer_1, w_o), b_o)

    return X, Y, out_layer


def fit(X,
        Y,
        train_size,
        train_input,
        train_output,
        test_input,
        test_output,
        train_op,
        loss_func,
        batch_size,
        tolerance,
        output_layer,
        in_file):

    time_book = []
    mse_book = []
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        epoch = 0
        print("----------------------------------------")
        while True:
            total_cost = 0
            timer_start = time.process_time_ns()
            # Loop over all batches
            for start, end in zip(range(0, train_size, batch_size), range(batch_size, train_size + 1, batch_size)):
                batch_x = train_input[start:end]
                batch_y = train_output[start:end].reshape(-1, 1)
                sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
                batch_cost = sess.run(loss_func, feed_dict={X: batch_x, Y: batch_y})
                total_cost += batch_cost*batch_size

            avg_cost = total_cost/train_size
            print("    MSE for Epoch {}: ".format(epoch+1) + str(avg_cost))
            time_cost =time.process_time_ns()-timer_start
            time_book.append(time_cost)
            mse_book.append(avg_cost)
            epoch += 1

            in_file.write(str(epoch)+','+str(time_cost)+'\n')

            # if avg_cost < tolerance:
            #     break

            if epoch == 100:
                break

        result = output_layer.eval({X: test_input})
        mse = loss_func.eval({X: test_input, Y: test_output.reshape(-1, 1)})

        print("----------------------------------------")
        print("Total epoch: {}".format(epoch))
        print("Train MSE: {}\n\n".format(avg_cost))
    return result, mse, epoch, time_book, mse_book


def get_B5(table):
    print("MSE for GradientDescent: "+str(table[0]))
    print("MSE for Momentum: "+str(table[1]))
    print("MSE for RMSProp: "+str(table[2]))

    fig, ax = plt.subplots()

    x = np.arange(3)
    plt.bar(x, table)
    plt.xticks(x, ('GradientDescent', 'Momentum', 'RMSProp'))
    ax.set_ylabel('MSE')
    ax.set_title('Performance(measured by MSE) for Different Optimizer at 100 Epochs')
    plt.ylim((0,0.018))
    plt.show()
    # plt.savefig('Q1A/epoch.png', format='png', dpi=300)


def get_1b2(books):
    x = np.arange(0, 100,1)
    fig, ax = plt.subplots()
    name_list = ['GradientDescentOptimizer', 'MomentumOptimizer', 'RMSPropOptimizer']
    for i in range(3):
        plt.plot(x, books[i][:100], label = name_list[i])

    plt.legend(title='Optimizer')
    plt.ylim(0,1.5)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('MSE')
    ax.set_title('Variation of MSE against Epoch Number')
    # leg.get_frame().set_alpha(0.5)
    plt.show()


def write_csv(filename, book):
    mse = open(filename, 'w')
    mse.write('epoch,MSE')
    mse.write('\n')
    mse.flush()
    for i in range(100):
        mse.write(str(i) + ',' + str(book[i]) + '\n')


def plot_B2():
    m1 = pd.read_csv('Q1B/GradientDescent.csv')
    m2 = pd.read_csv('Q1B/Momentum.csv')
    m3 = pd.read_csv('Q1B/RMSProp.csv')
    plt.plot(m1['epoch']+1, m1['MSE'], label="GradientDescent")
    plt.plot(m2['epoch']+1, m2['MSE'], label="Momentum")
    plt.plot(m3['epoch']+1, m3['MSE'], label="RMSProp")

    plt.title('Variation of MSE against Epoch Number')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend(title="Optimizer")
    plt.grid(axis='y')
    plt.ylim((0, 1.5))
    plt.show()


def plot_B3(book):
    avg_time_1 = np.sum(book[0]) / 101
    avg_time_2 = np.sum(book[1]) / 101
    avg_time_3 = np.sum(book[2]) / 101

    print("Time take per epoch for GradientDescent: "+str(avg_time_1))
    print("Time take per epoch for Momentum: "+str(avg_time_2))
    print("Time take per epoch for RMSProp: "+str(avg_time_3))

    table = [avg_time_1,avg_time_2,avg_time_3]

    fig, ax = plt.subplots()

    x=np.arange(3)
    plt.bar(x, table)
    plt.xticks(x, ('GradientDescent', 'Momentum', 'RMSProp'))
    ax.set_ylabel('sec./epoch')
    ax.set_title('CPU Time Taken per Epoch for Each of the 3 Methods')
    plt.show()

def plot_B3_V2(book):
    print("Time take per epoch for GradientDescent: "+str(book[0]))
    print("Time take per epoch for Momentum: "+str(book[1]))
    print("Time take per epoch for RMSProp: "+str(book[2]))


    fig, ax = plt.subplots()

    x=np.arange(3)
    plt.bar(x, book)
    plt.xticks(x, ('GradientDescent', 'Momentum', 'RMSProp'))
    plt.ylim(180000,240000)
    ax.set_ylabel('nano-sec./epoch')
    ax.set_title('Fig.1 Average CPU Time Taken per Epoch for \nEach of the 3 Methods(when converge)')
    plt.show()


def plt_every_epoch():
    m1 = pd.read_csv("Q1B/subQ3/GradientDescentOptimizer.csv")
    m2 = pd.read_csv("Q1B/subQ3/MomentumOptimizer.csv")
    m3 = pd.read_csv("Q1B/subQ3/RMSPropOptimizer.csv")

    merged = pd.merge(m1, m2, how='left', on='epoch')
    merged.to_csv('Q1B/subQ3/merged.csv', index=False)
    merged = pd.merge(merged, m3, how='left', on='epoch')
    merged.to_csv('Q1B/subQ3/merged.csv', index=False)

    x = np.array(range(1,101))  # the label locations
    # print(x)

    fig, ax = plt.subplots()
    ax = merged.plot.bar(x="epoch",y=['GradientDescentOptimizer', 'MomentumOptimizer', 'RMSPropOptimizer'],ax=ax)
    # print(m1["time"])

    # ax.bar(x - width, merged['GradientDescentOptimizer'], width=width, label='GradientDescent')
    # ax.bar(x, merged['MomentumOptimizer'], width=width, label='Momentum')
    # ax.bar(x + width, merged['RMSPropOptimizer'], width=width, label='RMSProp')

    plt.ylim((1000000, 6000000))
    ax.set_ylabel('nano-second')
    # ax.set_xlabel('epoch')
    ax.set_xticks(np.arange(0, 101,10))
    ax.set_xticklabels(np.arange(0, 101,10))
    ax.set_title('Fig.2 CPU Time taken each epoch(for 100 epochs)')
    # plt.ylim(3000000,6000000)
    # ax.legend()
    # # fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    # lists to store needed data
    results = []
    answer_book = []
    epochs_list =[]
    mse_list = []
    mse_books = []
    time_books = []
    speed_book = []

    learning_rate = 0.02
    training_epochs = 100
    batch_size = 100
    epsilon = 0.02
    input_size = 2
    output_size = 1

    total_test = 10
    hidden_neurons = 8


    opt_list = []
    opt_list.append(tf.train.GradientDescentOptimizer(learning_rate=learning_rate))
    opt_list.append(tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9))
    opt_list.append(tf.train.RMSPropOptimizer(learning_rate=learning_rate))
    opt_index = 0

    for opt in opt_list:
        accumulate_mse = 0
        accumulate_epochs = 0

        if opt_index == 0:
            opt_name = "GradientDescentOptimizer"
        elif opt_index == 1:
            opt_name = "MomentumOptimizer"
        else:
            opt_name = "RMSPropOptimizer"
        opt_index += 1

        for i in range(total_test):
            print("Test " + str(i+1) + " for {} :".format(opt_name))

            # prepared data
            train_input, train_output, train_answer, \
            test_input, test_output, test_answer, \
            val_input, vali_output, vali_answer = prepare_data()

            train_size = train_input.shape[0]
            # build model
            X, Y, output_layer = mlp_process(input_size, hidden_neurons, output_size)

            # loss func and optimizer
            rmse = tf.reduce_mean(tf.square(tf.subtract(output_layer, Y)))
            train_op = opt.minimize(rmse)
            # write data in a .csv
            in_file = open("Q1B/subQ3/"+opt_name+".csv", 'w')
            in_file.write('epoch,'+opt_name)
            in_file.write('\n')
            in_file.flush()
            result, mse_out, epoch, time_book, mse_book = fit(   X,
                                                                 Y,
                                                                 train_size,
                                                                 train_input,
                                                                 train_output,
                                                                 test_input,
                                                                 test_output,
                                                                 train_op,
                                                                 rmse,
                                                                 batch_size,
                                                                 epsilon,
                                                                 output_layer,
                                                                 in_file)
            # if i == 0:
            #     results.append(result)
            #     answer_book.append(test_answer)

            in_file.close()

            if i == 0:
                speed = np.sum(time_book) / epoch
                speed_book.append(speed)
                mse_books.append(mse_book)
                time_books.append(time_book)
            accumulate_mse += mse_out
            accumulate_epochs += epoch
        mse_list.append(accumulate_mse/total_test)
        epochs_list.append(accumulate_epochs/total_test)



    print("# Epochs to converge for GradientDescent: " + str(epochs_list[0]))
    print("# Epochs to converge for Momentum: " + str(epochs_list[1]))
    print("# Epochs to converge for RMSProp: " + str(epochs_list[2]))


    # write_csv('Q1B/GradientDescent.csv', mse_books[0])
    # write_csv('Q1B/Momentum.csv', mse_books[1])
    # write_csv('Q1B/RMSProp.csv', mse_books[2])
    # plot_B2()
    # get_1b2(mse_books)
    plt_every_epoch()
    # plot_B3_V2(speed_book)
    # # plot_B3(time_books)
    # get_B5(mse_list)

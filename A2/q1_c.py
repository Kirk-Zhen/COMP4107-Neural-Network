import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.patches as mpatches

def fit_early_stop( X,
                    Y,
                    train_size,
                    train_input,
                    train_output,
                    test_input,
                    test_output,
                    vali_input,
                    vali_output,
                    train_op,
                    loss_func,
                    batch_size,
                    tolerance,
                    output_layer):


    rmse_func = tf.sqrt(loss_func)
    flag = True
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        train_book=[]
        test_book =[]
        vali_book = []
        epoch = 0
        pre_vali_mse = 100
        count = 0

        print("----------------------------------------")
        while True:
            total_cost = 0
            for start, end in zip(range(0, train_size, batch_size), range(batch_size, train_size + 1, batch_size)):
                batch_x = train_input[start:end]
                batch_y = train_output[start:end].reshape(-1, 1)
                # Run optimization op (backprop) and cost op (to get loss value)
                sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
                batch_cost = sess.run(loss_func, feed_dict={X: batch_x, Y: batch_y})
                total_cost += batch_cost*batch_size

            avg_cost = total_cost/train_size
            rmse = rmse_func.eval({X: test_input, Y: test_output.reshape(-1, 1)})
            test_mse = loss_func.eval({X: test_input, Y: test_output.reshape(-1, 1)})
            vali_mse = loss_func.eval({X: vali_input, Y: vali_output.reshape(-1, 1)})

            if epoch % 100 == 0:
                print("    Cost for Epoch {}: ".format(epoch+1) + str(rmse))

            epoch += 1
            if epoch % 1 == 0:
                if vali_mse > pre_vali_mse:
                    count = count + 1
                    # print("rmse increased " + str(count) + " times")
                    if count > 10:
                        print("break count")
                        break
                else:
                    count = 0


            pre_vali_mse = vali_mse

            if rmse < tolerance:
                print("break RMSE")
                break

            # if avg_cost < tolerance:
            #     print("break MSE")
            #     break
            train_book.append(avg_cost)
            test_book.append(test_mse)
            vali_book.append(vali_mse)

            if avg_cost < 0.0025 and flag:
                con_epoch = epoch
                flag = False


        result = output_layer.eval({X: test_input})
        mse = loss_func.eval({X: test_input, Y: test_output.reshape(-1, 1)})

        print("----------------------------------------")
        print("Total epoch: {}".format(epoch))
        print("Test RMSE: {}\n\n".format(rmse))
    return train_book, test_book, vali_book, result, mse, con_epoch


def show_function_surface(x, y):
    X,Y = np.meshgrid(x, y)

    Z = f(X,Y)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    ax.set_zlim(-1, 1)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


def show_simple_contour(x, y):
    X, Y = np.meshgrid(x, y)

    Z = f(X, Y)
    z = f(X, Y)
    fig = plt.figure()
    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, z)
    ax.clabel(CS, CS.levels, inline=True, fontsize=10)
    plt.show()


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
    vali_input, vali_output, vali_answer = shuffle_with_answer(vali_mtx, vali_output)

    return train_input, train_output, test_answer, test_input, test_output,test_answer, vali_input, vali_output, vali_answer


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
        output_layer):
    func_r = tf.sqrt(loss_func)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        epoch = 0
        print("----------------------------------------")
        while True:
            total_cost = 0
            for start, end in zip(range(0, train_size, batch_size), range(batch_size, train_size + 1, batch_size)):
                batch_x = train_input[start:end]
                batch_y = train_output[start:end].reshape(-1, 1)
                # Run optimization op (backprop) and cost op (to get loss value)
                sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
                batch_cost = sess.run(loss_func, feed_dict={X: batch_x, Y: batch_y})
                total_cost += batch_cost*batch_size

            avg_cost = total_cost/train_size
            cost = func_r.eval({X: train_input, Y: train_output.reshape(-1, 1)})

            if epoch % 100 == 0:
                print("    Cost for Epoch {}: ".format(epoch+1) + str(cost))

            epoch += 1
            if cost < tolerance:
                print("break!!!")
                break



        result = output_layer.eval({X: test_input})
        mse = loss_func.eval({X: test_input, Y: test_output.reshape(-1, 1)})

        print("----------------------------------------")
        print("Total epoch: {}".format(epoch))
        print("Train RMSE: {}\n\n".format(rmse))
    return result, mse, epoch


def plot_converge_performance(table):
    print("MSE for 2 Hidden Neurons: "+str(table[0]))
    print("MSE for 8 Hidden Neurons: "+str(table[1]))
    print("MSE for 50 Hidden Neurons: "+str(table[2]))

    fig, ax = plt.subplots()

    x = np.arange(3)
    plt.bar(x, table)
    plt.xticks(x, ('2 Neurons', '8 Neurons', '50 Neurons'))
    ax.set_ylabel('MSE')
    ax.set_title('Influence of # Hidden Layer Neurons on Performance(MSE)')
    plt.ylim(0, 0.02)
    plt.show()
    # plt.savefig('Q1A/epoch.png', format='png', dpi=300)


def plot_converge_epoch(table):
    print("# Epochs to converge for 2 Hidden Neurons: "+str(table[0]))
    print("# Epochs to converge for 8 Hidden Neurons: "+str(table[1]))
    print("# Epochs to converge for 50 Hidden Neurons: "+str(table[2]))

    fig, ax = plt.subplots()

    x=np.arange(3)
    plt.bar(x, table)
    plt.xticks(x, ('2 Neurons', '8 Neurons', '50 Neurons'))
    ax.set_ylabel('Epochs')
    ax.set_title('Influence of # Hidden Layer Neurons on \n# Epochs to converge')
    plt.show()
    # plt.savefig('Q1A/epoch.png', format = 'png', dpi = 300)

def plot_converge_V2(epoch_list):
    fig, ax = plt.subplots()

    x = np.arange(5)
    plt.bar(x, epoch_list)
    plt.xticks(x, ('8 Neurons', '9 Neurons', '10 Neurons','11 Neurons', '12 Neurons'))
    ax.set_ylabel('Epochs')
    ax.set_title('Fig.2 Epochs to Converge for Different Hidden Neurons Size')
    plt.show()


def plot_fig_7(results, book):
    test_scaler = np.arange(-1 + 1 / 9, 1, 2 / 9)
    test_X, test_Y = np.meshgrid(test_scaler, test_scaler)
    test_Z = f(test_X, test_Y)

    n_2 = []
    n_8 = []
    # n_50 = []
    for i in range(0, 81):
        po_2 = np.where(book[0] == i)[0][0]
        po_8 = np.where(book[1] == i)[0][0]
        # po_50 = np.where(book[2] == i)[0][0]
        n_2.append(results[0][po_2])
        n_8.append(results[1][po_8])
        # n_50.append(results[2][po_50])
    n_2=np.array(n_2)
    n_8 = np.array(n_8)
    # n_50 = np.array(n_50)

    plt.figure()
    fig, ax = plt.subplots()
    CS = ax.contour(test_X, test_Y, test_Z, colors='black', linewidths=2)
    CS1 = ax.contour(test_X, test_Y, np.reshape(n_2, (9,9)).T, colors='red')
    CS2 = ax.contour(test_X, test_Y, np.reshape(n_8, (9,9)).T, colors='lightgreen')
    # CS3 = ax.contour(test_X, test_Y, np.reshape(n_50, (9,9)).T, colors='blue')

    black_patch = mpatches.Patch(color='black', label='target')
    red_patch = mpatches.Patch(color='red', label='without early stopping')
    green_patch = mpatches.Patch(color='lightgreen', label='with early stopping')
    # blue_patch = mpatches.Patch(color='blue', label='50 neurons')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), handles=[black_patch, green_patch, red_patch])
    # plt.legend( handles=[black_patch, green_patch, red_patch])
    ax.set_title('Contours with/without early stopping')
    plt.show()



def plot_diff_neurons(mse_list, neuron_list):
    # for i in range(len(mse_list)):
    #     print("MSE for {} Hidden Neurons: ".format(neuron_list[i]) + str(mse_list[i]))

    plt.subplots()
    plt.plot(neuron_list, mse_list, label="Average MSE")

    plt.title('Fig.1 MSE at Convergence against Hidden Layer Size')
    plt.xlabel('Hidden Layer Size')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(axis='y')
    plt.ylim()
    plt.show()

def plot_diff_convergence(epoch_list, neuron_list):
    plt.subplots()
    plt.plot(neuron_list, epoch_list, label="Average Epoch to Converge")

    plt.title('Fig.2 Epochs to Convergence against Hidden Layer Size')
    plt.xlabel('Hidden Layer Size')
    plt.ylabel('Epoch')
    plt.legend()
    plt.grid(axis='y')
    plt.ylim()
    plt.show()


def plot_fig_6(train_book, test_book, vali_book, epoch):
    plt.subplots()
    plt.plot([epoch - 1, epoch, epoch + 1], train_book[(epoch-1): (epoch + 2)], label="training")
    plt.plot([epoch - 1, epoch, epoch + 1], test_book[(epoch-1): (epoch + 2)], label="test")
    plt.plot([epoch - 1, epoch, epoch + 1], vali_book[(epoch-1): (epoch + 2)], label="validation")
    plt.plot([epoch - 1, epoch, epoch + 1], [0.0025,0.0025,0.0025], label="goal")

    # plt.plot([epoch - 1, epoch, epoch + 1], [0.0025,0.0025,0.0025], label="training")
    # plt.plot([epoch - 1, epoch, epoch + 1], [0.0025,0.0025,0.0025], label="test")
    # plt.plot([epoch - 1, epoch, epoch + 1], [0.0025,0.0025,0.0025], label="validation")
    # plt.plot([epoch - 1, epoch, epoch + 1], [0.0025,0.0025,0.0025], label="goal")

    plt.title('Evaluation of MSE')
    # plt.ylabel('')
    plt.xlabel('2 Epoch')
    plt.legend()
    # plt.grid(axis='y')
    plt.xticks(range(epoch - 1, epoch + 2, 1))
    plt.ylim()
    plt.show()



if __name__ == '__main__':
    results = []
    answer_book = []
    epochs_list =[]
    mse_list = []

    total_test = 10
    # hidden_list = list(range(2,51,2))
    hidden_list = [8]
    for hidden_neurons in hidden_list:
        accumulate_mse = 0
        accumulate_epochs = 0
        for i in range(total_test):
            print("Test " + str(i+1) + " for {} Hidden Neurons:".format(hidden_neurons))

            # get data prepared
            train_input, train_output, train_answer, \
            test_input, test_output, test_answer, \
            vali_input, vali_output, vali_answer = prepare_data()

            train_size = train_input.shape[0]
            learning_rate = 0.001
            batch_size = 100
            epsilon = 0.02
            input_size = 2
            output_size = 1

            # build model
            X, Y, output_layer = mlp_process(input_size, hidden_neurons, output_size)

            # initiate loss func and optimizer
            rmse = tf.reduce_mean(tf.square(tf.subtract(output_layer, Y)))
            optimizer = tf.train.RMSPropOptimizer(learning_rate)
            train_op = optimizer.minimize(rmse)

            # fit the model and get data
            result, _, _ = fit( X,
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
                                output_layer)
            results.append(result)
            answer_book.append(test_answer)

            train_book, test_book, vali_book,\
            result, _, _ = fit_early_stop(  X,
                                            Y,
                                            train_size,
                                            train_input,
                                            train_output,
                                            test_input,
                                            test_output,
                                            vali_input,
                                            vali_output,
                                            train_op,
                                            rmse,
                                            batch_size,
                                            epsilon,
                                            output_layer)
            results.append(result)
            answer_book.append(test_answer)


            # aaaaa =train_book[(epoch-1): (epoch + 2)]
            # plot_fig_6(train_book, test_book, vali_book, epoch)
            # accumulate_mse += mse_out
            # accumulate_epochs += epoch
        # mse_list.append(accumulate_mse/total_test)
        # epochs_list.append(accumulate_epochs/total_test)

    plot_fig_7(results, answer_book)
    # plot_converge_performance(mse_list)
    # plot_converge_epoch(epochs_list)
    # plot_contour(results, answer_book)
    # plot_diff_neurons(mse_list,hidden_list)
    # plot_diff_convergence(epochs_list, hidden_list)
    # plot_converge_V2(hidden_list)

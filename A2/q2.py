import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

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
    layer_1 = tf.nn.sigmoid(layer_1)
    out_layer = tf.add(tf.matmul(layer_1, w_o), b_o)
    out_layer = tf.nn.softmax(out_layer)
    return X, Y, out_layer


def shuffle_with_answer(mtx, label):
    answer = np.linspace(0, mtx.shape[0]-1, mtx.shape[0], dtype=int)
    np.random.shuffle(answer)
    return mtx[answer], label[answer], answer


def prepare_data(noise_level):
    infile = open('Q2/q2_kirk.txt')
    raw_mtx = infile.read()
    vector_list = raw_mtx.split('\n')
    ideal_input = []
    ideal_output = []
    for count in range(len(vector_list)):
        vector = []
        for var in vector_list[count]:
            vector.append(int(var))
        ideal_input.append(np.array(vector))
        label_vector = np.zeros(31)
        label_vector[count] = 1
        ideal_output.append(np.array(label_vector))

    ideal_input, ideal_output, answer = shuffle_with_answer(np.array(ideal_input), np.array(ideal_output))
    noisy_input = np.copy(ideal_input)
    noisy_label = np.copy(ideal_output)
    for row in noisy_input:
        noise_index = random.sample(range(0, row.shape[0]), noise_level)
        for index in noise_index:
            if row[index] == 1:
                row[index] = 0
            elif row[index] == 0:
                row[index] = 1
    test_input = np.concatenate((ideal_input, noisy_input), axis=0)
    test_output = np.concatenate((ideal_output, noisy_label), axis=0)
    test_input, test_output, _ = shuffle_with_answer(test_input, test_output)

    return ideal_input, ideal_output, noisy_input, noisy_label, test_input, test_output


def get_accuracy(test_input, test_output):
    predict = output_layer
    check_correct = tf.equal(tf.argmax(predict, 1), tf.argmax(Y, 1))
    acc_func = tf.reduce_mean(tf.cast(check_correct, "float"))
    acc_rate = acc_func.eval({X: test_input, Y: test_output})
    return acc_rate


def plot_q2_A(book):
    noise_list = [0,1,2,3]
    m_05 = book[0]
    m_10 = book[1]
    m_15 = book[2]
    m_20 = book[3]
    m_25 = book[4]

    plt.plot(noise_list, m_05, label="05")
    plt.plot(noise_list, m_10, label="10")
    plt.plot(noise_list, m_15, label="15")
    plt.plot(noise_list, m_20, label="20")
    plt.plot(noise_list, m_25, label="25")

    plt.title('Fig.1 Recognition Error of different Hidden Layer Size')
    plt.xlabel('Noise Level')
    plt.ylabel('Percentage of Recognition Error')
    plt.legend(title="Hidden Neurons")
    plt.grid(axis='y')
    # plt.ylim()
    plt.show()

def plot_q2_b(book, title):
    plt.plot(book, label="Loss")
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss Function')
    plt.yscale('log')
    plt.show()


def plot_q2_C(with_book,without_book):
    noise_list = [0,1,2,3]

    plt.plot(noise_list, without_book, 'k--',label="trained without noise")
    plt.plot(noise_list, with_book, label="trained with noise", color='red')

    plt.title('Recognition Error with Different Training Strategy')
    plt.xlabel('Noise Level')
    plt.ylabel('Percentage of Recognition Error')
    plt.legend(title="Training Strategy")
    plt.grid(axis='y')
    plt.show()

if __name__ == '__main__':

    learning_rate = 0.002
    input_size = 35
    output_size = 31

    # the parameter now is for Q2A
    # range_hidden = [5, 10, 15, 20, 25]
    # noise_levels = [0, 1, 2, 3]
    # process_trigger = [1]
    # question = "A"

    # for Q2B, please comment above 3 lists and use:
    range_hidden = [15]
    noise_levels = [3]
    process_trigger = [1]
    question = "B"

    # for Q2C, please comment above 3 lists and use:
    # range_hidden = [15]
    # noise_levels = [0, 1, 2, 3]
    # process_trigger = [1, 2]
    # question = "C"

    #  You should also choose the corresponding function in the main

    # initiate lists to store values
    first_step_book=[]
    final_step_book=[]
    hidden_error_list = []
    without_noise_list=[]
    with_noise_list = []
    for process in process_trigger:
        for hidden_neurons in range_hidden:
            nosie_error_list = []
            for noise_level in noise_levels:
                X, Y, output_layer = mlp_process(input_size, hidden_neurons, output_size)
                loss_func = tf.reduce_mean(
                    tf.square(tf.subtract(Y, output_layer)))
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                train_op = optimizer.minimize(loss_func)

                init = tf.global_variables_initializer()
                with tf.Session() as sess:
                    sess.run(init)

                    if process == 1:
                        step_nums = [1, 2, 3]
                    else:
                        step_nums = [1, 3]
                    for step in step_nums:
                        print("======================Evaluating==========================")
                        print("Step " + str(step) + ", Size " + str(hidden_neurons) + ", Error " + str(noise_level))
                        epoch = 0
                        ideal_input, ideal_label, \
                        noisy_input, noisy_label, \
                        test_input, test_label = prepare_data(noise_level)

                        if step == 2:
                            train_input = noisy_input
                            train_label = noisy_label
                        else:
                            train_input = ideal_input
                            train_label = ideal_label
                        train_size = train_input.shape[0]
                        while True:
                            # total_cost = 0
                            avg_cost = 0
                            for i in range(train_size):
                                batch_x = train_input
                                batch_y = train_label
                                sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
                                batch_cost = sess.run(loss_func, feed_dict={X: batch_x, Y: batch_y})
                                avg_cost += batch_cost / train_size

                            if epoch % 20 == 0:
                                print("Cost for Epoch {}: ".format(epoch + 1) + str(avg_cost))

                            if question == "B":
                                temp = 0.000001
                            else:
                                temp = 0.01

                            if step == 2:
                                if avg_cost <= 0.01:
                                    break
                            else:
                                if step == 1:
                                    first_step_book.append(avg_cost)
                                else:  # step == 3
                                    final_step_book.append(avg_cost)
                                if avg_cost <= temp:
                                    acc_rate = get_accuracy(batch_x, batch_y)
                                    if acc_rate == 1:
                                        break
                            epoch += 1

                    _, _, _, _, test_input, test_label = prepare_data(noise_level)
                    acc_rate = get_accuracy(test_input, test_label)
                    print("Accuracy: " + str(acc_rate))
                    reg_error = 1 - acc_rate
                    print("Recognition Errors:" + str(reg_error))
                if process == 1:
                    with_noise_list.append(reg_error)
                else:
                    without_noise_list.append(reg_error)
                nosie_error_list.append(reg_error)

            hidden_error_list.append(nosie_error_list)

    # if you want to call some of these functions, please review the inline comments above to
    # comment or uncomment some line of code to enable these functions.

    # plot_q2_A(hidden_error_list)
    plot_q2_b(first_step_book, "First Training Step")
    plot_q2_b(final_step_book, "Final Training Step")
    # plot_q2_C(with_noise_list, without_noise_list)

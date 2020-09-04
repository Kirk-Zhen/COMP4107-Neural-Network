import pandas as pd, numpy as np, scipy as sp
from matplotlib import pyplot as plt



def get_figure(in_path, out_path, epsilon):
    df = pd.read_csv(in_path)
    plt.subplots()
    title = 'Graph for epsilon = {}'.format(epsilon)
    x1 = df['iteration']
    y1 = df['sqr']
    plt.plot(x1, y1, marker='', color='red', markeredgecolor='blue', label='Least Square')
    plt.title(title)
    plt.ylabel('Least Square')
    plt.xlabel('Number of Iterations')
    plt.grid(axis='y')
    plt.legend(loc='best')
    plt.savefig(out_path, format='png', dpi=1200)


A = np.array(
    [
        [1, 2, 3],
        [2, 3, 4],
        [4, 5, 6],
        [1, 1, 1],
    ]
)

b = np.array(
    [1, 1, 1, 1]
)

step_sizes = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5]
tolerance = 0.01
num_ep = 0
for epsilon in step_sizes:
    print('Epsilon = ', epsilon)
    x = np.array([0, 0, 0])
    iteration = 0

    # sqr = open('data/sqr{}.csv'.format(num_ep), 'w')
    # sqr.write('iteration,sqr')
    # sqr.write('\n')
    # sqr.flush()
    while np.linalg.norm((np.linalg.multi_dot([A.T, A, x]) - np.dot(A.T, b))) > tolerance:
        with np.errstate(invalid='ignore'):
            x = x - epsilon*(np.linalg.multi_dot([A.T, A, x]) - np.dot(A.T, b))
        iteration += 1
        # sqr.write(str(iteration) + ',' +
        #           str((1/2)*np.linalg.norm((np.linalg.multi_dot([A.T, A, x]) - np.dot(A.T, b)))) +
        #           '\n')

    # get_figure('data/sqr{}.csv'.format(num_ep), 'q3_figure/figure{}.jpg'.format(num_ep), epsilon)

    print('x = ' + str(x))
    flag = True
    for ele in x:
        if np.isnan(ele):
            flag = False
    if flag:
        print('Result Converge')
    else:
        print('Result Diverge')
    print('Total Iterations: ' + str(iteration)+'\n')
    num_ep += 1

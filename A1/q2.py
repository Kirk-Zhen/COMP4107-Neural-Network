import numpy as np

# create grid space, calculate value z, and store values in matrix A
# Refer to https://docs.scipy.org/doc/numpy/reference/generated/numpy.meshgrid.html?highlight=meshgrid#numpy.meshgrid
x = np.arange(-0.7, 0.701, 0.001)
y = np.arange(-0.7, 0.701, 0.001)
xx, yy = np.meshgrid(x, y, sparse=False)
# calculate z for each x_i and y_j, and store in matrix A
A = np.sqrt(1 - xx**2 - yy**2)


# print out matrix A
print("A: \n" + str(A))

# compute SVD for matrix A
U, s, V = np.linalg.svd(A)
# transfer s into a diagonal matrix
S = np.diag(s)

# compute the best rank(2) approximation of A
A_2 = np.linalg.multi_dot([U[:, :2], S[:2, :2], V[:2, :]])
print("\nRank(2) approximation of A: \n" + str(A_2))

# compute the norm of A - A_2
print("\n||A-A_2|| in order 1: " + str(np.linalg.norm(A-A_2, 1)))
print("||A-A_2|| in order 2: " + str(np.linalg.norm(A-A_2, 2)))

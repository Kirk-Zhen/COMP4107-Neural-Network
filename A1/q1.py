import numpy as np

A = np.array(
    [
        [1, 2, 3],
        [2, 3, 4],
        [4, 5, 6],
        [1, 1, 1],
    ]
)

U, s, V = np.linalg.svd(A)

# print out the result calculated by numpy
print("U: \n" + str(U))
print("s: \n" + str(s))
print("Transpose of V: \n" + str(V))

# transfer s into a diagonal matrix
S = np.diag(s)
print("\nTransfer s in to a diagonal matrix S:")
print("S: \n" + str(S))

# Take U_3 as first part of SVD
U_3 = U[:, :3]
print("\nTake U_3 as first part of SVD")
print("U_3: \n" + str(U_3))

# Reproduce matrix A
print("\nReproduce matrix A use U_3, S, V^T:\n" +
      str(np.linalg.multi_dot([U[:, :3], np.diag(s), V])))

# another way to reproduce matrix A
# print("\nsvd leads to matrix A:\n" +
#       str(np.linalg.multi_dot([u, np.diag(np.append(s, 0))[:, :3], v])))

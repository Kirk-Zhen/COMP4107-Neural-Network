import numpy as np
from scipy import linalg as sp

A = np.array(
    [
        [3, 2, -1, 4],
        [1, 0, 2, 3],
        [-2, -2, 3, -1]
    ]
)

# calculate the null space of matrix A
A_null = sp.null_space(A)
print("The null space of matrix A is:\n"+str(A_null))

# get 1st and 2nd column of the null space of A
a_1 = A_null[:, 0]
a_2 = A_null[:, 1]
print("\nThe two independent vectors in null space of A are:\n"+str(a_1)+" and "+str(a_2))


# Get the values that we need to determine whether cols and rows are independent
# Determine the number of columns and rows in A
cols = np.size(A, 1)
rows = np.size(A, 0)
# Determine rank of matrix A
rank_A = np.linalg.matrix_rank(A)
# rank of A transpose
rank_AT = np.linalg.matrix_rank(A.T)


print("\nNumber of columns in matrix A: "+str(cols))
print("Rank(A): "+str(rank_A))
# Compare Rank(A) and number of cols, to determine whether columns are independent
if cols > rank_A:
    print("Columns of A are not linearly independent")
else:
    print("Columns of A are linearly independent")

print("\nNumber of rows in matrix A: "+str(rows))
print("Rank(Transpose of A): "+str(rank_AT))
# Compare Rank(AT) and number of rows, to determine whether rows are independent
if rows > rank_AT:
    print("Rows of A are not linearly independent")
else:
    print("Rows of A are linearly independent")

# Determine Moore-Penrose Pseudoinverse of Matrix A
A_inverse = np.linalg.pinv(A)

print("\nMoore-Penrose Pseudoinverse of Matrix A:\n" + str(A_inverse))

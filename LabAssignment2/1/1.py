import numpy as np

# A. Create a 1d array M with values ranging from 5 to 21 and print M.
M = np.arange(5, 21)
print(M)

# B. Reshape M as a 4x4 matrix and print M.
M = M.reshape(4, 4)
print(M)

# C. Set the value of “inner” elements of the matrix M to 0 and print M.
M[1:-1, 1:-1] = 0
print(M)

# D. Assign M^2 to the M and print M.
M = M @ M
print(M)

# E. Let’s call the first row of the matrix M a vector v.
# Calculate the magnitude of the vector v and print it.
#   i. Hint:
#   ii. Hint: Use np.sqrt()
v = M[0]
print(np.sqrt(np.sum(np.square(v))))


# F. Files to submit: A Python source file
# (Name the file whatever you want (in English). Extension should be .py))
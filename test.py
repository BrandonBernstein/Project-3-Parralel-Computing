import numpy as np

# Set up parameters
N = 8  # Size of the matrix (N x N)
size = 4  # Number of processes, should be a perfect square
procDim = int(np.sqrt(size))  # Dimensions of processor grid (2x2 if size=4)
blockDim = N // procDim  # Dimension of each submatrix

# Create an example N x N matrix
A = np.random.uniform(-1, 1, (N, N))
print("Matrix A:\n", A)

# Split A into procDim x procDim grid of submatrices
submatrices = [
    A[i*blockDim:(i+1)*blockDim, j*blockDim:(j+1)*blockDim]
    for i in range(procDim)
    for j in range(procDim)
]

# Each submatrix is now in submatrices[i] for i = 0, ..., size-1
for idx, submat in enumerate(submatrices):
    print(f"Submatrix {idx}:\n{submat}\n")
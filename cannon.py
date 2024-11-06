import numpy as np
from mpi4py import MPI
import sys

def cannon():
    N = int(sys.argv[1])
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()


    if rank == 0:
        sqroot = int(size**0.5)

        procDim = int(sqroot)
        blockDim = int(N / procDim)

        bCastInfo = np.array([procDim, blockDim, N], dtype=np.int32)
    else:
        # Ensure other processes have the same type and shape for bCastInfo
        bCastInfo = np.empty(3, dtype=np.int32)

    comm.Bcast(bCastInfo,root = 0)

    procDim = int(bCastInfo[0])
    blockDim = int(bCastInfo[1])
    N = int(bCastInfo[2])

    localA = np.empty((blockDim, blockDim), dtype='float64')
    localB = np.empty((blockDim, blockDim), dtype='float64')
    localC = np.zeros((blockDim, blockDim), dtype='float64')

    dims = MPI.Compute_dims(size, 2)
    cart_comm = comm.Create_cart(dims = dims, periods = [True,True], reorder = True)

    rank = comm.Get_rank() # Resetting rank.

    if rank == 0:
        # Generate two random matrices of size N x N
        A = np.random.uniform(-1, 1, (N, N))
        B = np.random.uniform(-1, 1, (N, N))
        # print("Matrix A:\n", A)
        # print("Matrix B:\n", B)

        # Split A and B into contiguous submatrices
        submatricesA = [
            np.ascontiguousarray(A[i*blockDim:(i+1)*blockDim, j*blockDim:(j+1)*blockDim])
            for i in range(procDim)
            for j in range(procDim)
        ]
        submatricesB = [
            np.ascontiguousarray(B[i*blockDim:(i+1)*blockDim, j*blockDim:(j+1)*blockDim])
            for i in range(procDim)
            for j in range(procDim)
        ]
    else:
        submatricesA = None
        submatricesB = None

    # Send submatrices based on Cartesian coordinates
    if rank == 0:
        for i in range(procDim):
            for j in range(procDim):
                # Calculate the target rank from the Cartesian coordinates
                target_rank = cart_comm.Get_cart_rank([i, j])
                if target_rank != 0:
                    # Send the appropriate submatrix based on coordinates
                    comm.Send(submatricesA[i * procDim + j], dest=target_rank, tag=0)
                    comm.Send(submatricesB[i * procDim + j], dest=target_rank, tag=1)
        # Root keeps its own submatrices
        localA = submatricesA[0]
        localB = submatricesB[0]
    else:
        # Other processes receive their submatrices based on Cartesian ranks
        comm.Recv(localA, source=0, tag=0)
        comm.Recv(localB, source=0, tag=1)



    coords = cart_comm.Get_coords(rank)
    cart_rank = cart_comm.Get_rank()  # Get the Cartesian rank (may be different from linear rank)


    left, right = cart_comm.Shift(1, 1)  # Shift left by row coordinate
    cart_comm.Sendrecv_replace(localA, dest=left, sendtag=1, source=right, recvtag=1)

    up, down = cart_comm.Shift(0, 1)  # Shift up by column coordinate
    cart_comm.Sendrecv_replace(localB, dest=up, sendtag=2, source=down, recvtag=2)


    start_time = MPI.Wtime()

    for i in range(0, procDim):
        localC += np.dot(localA, localB)
        # print("Local C:\n",localC)
        left, right = cart_comm.Shift(1, 1)
        up, down = cart_comm.Shift(0, 1)

        cart_comm.Sendrecv_replace(localA, dest=left, sendtag=1, source=right, recvtag=1)
        cart_comm.Sendrecv_replace(localB, dest=up, sendtag=2, source=down, recvtag=2)

    globalC = None
    if rank == 0:
        globalC = np.empty((N, N), dtype='float64')

    comm.Gatherv(
        sendbuf=localC,
        recvbuf=(globalC, blockDim**2),
        root=0
    )

    comm.Barrier()

    end_time = MPI.Wtime()

    if rank == 0:
        print(f"Time taken on: P = {size}, N = {N} was {end_time - start_time}")




if __name__ == "__main__":
    cannon()


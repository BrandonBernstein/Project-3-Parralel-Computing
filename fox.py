from mpi4py import MPI
import numpy as np
import sys


def fox():
    np.set_printoptions(edgeitems=30, linewidth=100000,
                        formatter=dict(float=lambda x: "%.3g" % x))

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Read matrix dimension N from command line argument
    N = int(sys.argv[1])

    # Broadcast grid dimensions and block size
    if rank == 0:
        sqroot = int(size ** 0.5)
        procDim = sqroot
        blockDim = N // procDim
        bCastInfo = np.array([procDim, blockDim, N], dtype=np.int32)
    else:
        bCastInfo = np.empty(3, dtype=np.int32)

    comm.Bcast(bCastInfo, root=0)
    procDim, blockDim, N = bCastInfo

    # Initialize sub-matrices for local computation
    localA = np.empty((blockDim, blockDim), dtype='float64')
    localB = np.empty((blockDim, blockDim), dtype='float64')
    localC = np.zeros((blockDim, blockDim), dtype='float64')

    # Set up Cartesian topology
    dims = [procDim, procDim]
    cart_comm = comm.Create_cart(dims=dims, periods=[True, True], reorder=True)
    coords = cart_comm.Get_coords(rank)
    row_rank, col_rank = coords

    if rank == 0:
        # Generate two random matrices of size N x N
        A = np.random.uniform(-1, 1, (N, N))
        B = np.random.uniform(-1, 1, (N, N))

        # Distribute full submatrix grid for A
        submatricesA = [
            np.ascontiguousarray(A[i * blockDim:(i + 1) * blockDim, j * blockDim:(j + 1) * blockDim])
            for i in range(procDim)
            for j in range(procDim)
        ]

        submatricesB = [
            np.ascontiguousarray(B[i * blockDim:(i + 1) * blockDim, j * blockDim:(j + 1) * blockDim])
            for i in range(procDim)
            for j in range(procDim)
        ]
    else:
        submatricesA = None
        submatricesB = None

    if rank == 0:
        print("Reached point 0")

    comm.Barrier()

    # Distribute B submatrices
    if rank == 0:
        for i in range(procDim):
            for j in range(procDim):
                target_rank = cart_comm.Get_cart_rank([i, j])
                if target_rank != 0:
                    comm.Send(submatricesB[i * procDim + j], dest=target_rank, tag=1)
        localB = submatricesB[0]  # Root keeps its own submatrix
    else:
        localB = np.empty((blockDim, blockDim), dtype='float64')
        comm.Recv(localB, source=0, tag=1)

    if rank == 0:
        print("Reached point 2")
    comm.Barrier()
    # Distribute A submatrices
    submatricesA = comm.bcast(submatricesA, root=0)
    # Fox algorithm main loop
    buff_A = np.empty_like(localA)  # Buffer for broadcasted A submatrices

    start_time = MPI.Wtime()

    for diag_stage in range(procDim):
        # Calculate which process is the "root" for this broadcast stage
        root = (row_rank + diag_stage) % procDim

        # Take appropriate submatrix of A
        localA = submatricesA[row_rank * procDim + root]

        # Each process multiplies its local A with local B
        localC += np.dot(localA, localB)

        # Shift the submatrix of B up in the column communicator
        up, down = cart_comm.Shift(0, 1)
        cart_comm.Sendrecv_replace(localB, dest=down, sendtag=diag_stage, source=up, recvtag=diag_stage)

    # Gather the results back to the root process

    if rank == 0:
        print("Reached point 3")

    globalC = None
    if rank == 0:
        globalC = np.empty((N, N), dtype='float64')

    comm.Barrier()
    comm.Gatherv(
        sendbuf=localC,
        recvbuf=(globalC, blockDim**2),
        root=0
    )
    comm.Barrier()
    end_time = MPI.Wtime()

    if rank == 0:
        print(f"Time taken on: P = {size}, N = {N} was {end_time - start_time}")


    if rank == 0:
        print(globalC)
        print(np.dot(A,B))

if __name__ == "__main__":
    fox()

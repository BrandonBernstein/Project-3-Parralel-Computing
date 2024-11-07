# Project 3: Parallel Matrix Multiplication

## Overview
This project focuses on implementing and analyzing two parallel algorithms for multiplying large square matrices in parallel: Cannon and Fox. Each algorithm will be briefly described, implemented, and checked for performance on varying matrix and processor sizes.

## Algorithms

**Cannon’s Algorithm:** A parallel matrix multiplication method designed for use on a $P \times P$ grid of processors, ideally suited for square matrices distributed across a mesh topology.

1. **Scatter** submatrices of size $\frac{N}{\sqrt{P}}$ to a grid of $\( \sqrt{P} \times \sqrt{P} \)$ processors.

2. **Initial Shifts**:
   - Shift submatrices of matrix $\( A \)$ to the left by one processor in the same row, with wrap-around.
   - Similarly, shift submatrices of matrix $\( B \)$ up by one processor in the same column, with wrap-around.

3. **Multiplication and Shifting**:
   - Perform matrix multiplication between submatrices on each processor for $\( \sqrt{P} \)$ stages.
   - After each multiplication, shift submatrices of $\( A \)$ left by one processor and submatrices of $\( B \) $up by one processor, both with wrap-around.

4. **Gather** results from all processors to form the final product matrix.

**Fox Algorithm:** An alternative approach for matrix multiplication on a mesh grid of processors. The Fox Algorithm is particularly suited for matrices distributed across a square grid and leverages a broadcast mechanism to optimize computation steps in parallel environments.

1. **Scatter** submatrices of $\( B \)$ with size $\( \frac{N}{\sqrt{P}} \)$ to a grid of $\( \sqrt{P} \times \sqrt{P} \)$ processors.

2. **Iteration over Stages**:
   - For diagonal stage $\( k = 0, 1, \dots, \sqrt{P} - 1 \)$:
     - **Broadcast Step**: 
       - The main diagonal processor on $row_i$ receives $row_i$ of the $\( k \)$-th diagonal submatrix of $\( A \)$, then broadcasts that submatrix to all processors in that row.
     - **Local Multiplication**:
       - Each processor performs local matrix multiplication between the received submatrix of $\( A \)$ and its submatrix of $\( B \)$.
     - **Shift Step**:
       - Each processor shifts its submatrix of $\( B \)$ up by one position in the same column, with wrap-around.

3. **Gather** results from all processors to form the final product matrix.

## Implementations
You can find Cannon's algorithm in cannon.py and Fox's method in fox.py. The proj3_run_script.txt contains the batch file used on seawulf, using the general run commands below:
   - Cannon’s Algorithm: mpirun -np <num_cores> python cannon.py <N>
   - Fox Algorithm: mpirun -np <num_cores> python fox.py <N>

The algorithms were run on processor sizes of 1,4,16, and 64 across matrix sizes (N x N) of 256, 1024, 4096 allowing me to document the following speed up curves.

![Cannon's Algorithm Speed up Curve](https://github.com/BrandonBernstein/Project-3-Parralel-Computing/blob/master/Cannon-Graph.png) ![Fox Method Speed up Curve](https://github.com/BrandonBernstein/Project-3-Parralel-Computing/blob/master/Fox-Graph.png)



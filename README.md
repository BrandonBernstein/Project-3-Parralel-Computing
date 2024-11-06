# Project 3: Parallel Matrix Multiplication
Due Date: November 10, 2024

Overview
This project focuses on implementing and analyzing two parallel algorithms for multiplying large square matrices on a parallel computer. The algorithms chosen are the Cannon’s Algorithm and the Fox Algorithm. Each algorithm is tested with matrices of various sizes and on systems with different numbers of CPU cores to assess performance and scalability.

Algorithms
Cannon’s Algorithm: This method is designed for distributed matrix multiplication, specifically suited for a mesh topology. Cannon’s Algorithm is efficient for minimizing inter-process communication, making it ideal for systems with high communication overheads.

Fox Algorithm: An alternative approach for matrix multiplication on a mesh grid of processors. The Fox Algorithm is particularly suited for matrices distributed across a square grid and leverages a broadcast mechanism to optimize computation steps in parallel environments.

Problem Description
Given two randomly generated matrices, A and B, with elements drawn uniformly from the range 
[
−
1
,
1
]
[−1,1], this project aims to:

Describe the matrix multiplication process using Cannon and Fox algorithms.
Implement both algorithms using mpi4py for parallel computing.
Test the performance of each algorithm on a parallel computer, varying:
Matrix sizes 
𝑁
=
2
8
,
2
10
,
2
12
N=2 
8
 ,2 
10
 ,2 
12
 
Number of CPU cores 
𝑃
=
2
2
,
2
4
,
2
6
P=2 
2
 ,2 
4
 ,2 
6
 
Collect and Analyze the timing results across different configurations.
Plot the speedup curves to visualize the performance improvements as the number of cores increases.
Comment on the performance results and discuss findings.
Implementation
The project uses mpi4py for parallel computing and leverages Python’s random number generator to create matrices with entries in the range 
[
−
1
,
1
]
[−1,1]. Both algorithms have been implemented to take advantage of parallel processing for efficient matrix multiplication.

Files
cannon.py: Contains the implementation of Cannon's Algorithm.
fox.py: Contains the implementation of Fox Algorithm.
generate_matrices.py: Script to generate random matrices of specified sizes.
performance_tests.py: Runs the matrix multiplication for specified matrix sizes and core counts, and gathers performance data.
plot_speedup.py: Generates speedup plots based on collected performance data.
Usage
Generate matrices: Run generate_matrices.py to create matrices with random entries in the range 
[
−
1
,
1
]
[−1,1].
Run algorithms:
Cannon’s Algorithm: mpiexec -n <num_cores> python cannon.py
Fox Algorithm: mpiexec -n <num_cores> python fox.py
Performance tests: Run performance_tests.py to automatically benchmark both algorithms for different matrix sizes and core counts.
Plot results: Use plot_speedup.py to create plots of speedup versus number of cores.
Results
The results section will analyze:

Execution Time: The time taken by each algorithm for different matrix sizes and core counts.
Speedup: The performance improvement observed as the number of cores increases.
Scalability: Insights into how each algorithm scales with matrix size and core count.
Conclusions
Based on the results, this project will conclude with observations on the relative performance of Cannon and Fox algorithms. The analysis will highlight the effectiveness of each algorithm under varying conditions and recommend optimal scenarios for their use in parallel matrix multiplication.

# Exercise 3: Jacobi's Algorithm with One-Sided MPI<!-- omit in toc -->

## Table of Contents <!-- omit in toc -->
- [Introduction](#introduction)
- [Jacobi's algorithm](#jacobis-algorithm)
- [Distribute the domain: MPI](#distribute-the-domain-mpi)
- [Results](#results)
  - [Save time](#save-time)
- [How to run](#how-to-run)
- [Check correctness](#check-correctness)
  
## Introduction

The third assignment consists of implementing the Jacobi's method to solve Laplace equation in a distributed memory environment, using the MPI library to communicate between processes in a one-sided fashion.

Before digging into the implementation of the algorithm, let's first describe the problem and how to solve it.

## Jacobi's algorithm

Laplace's equation is a second-order partial differential equation, often written in the form

$$
\nabla^2 V = 0
$$

where $V$ is the unknown function of the spatial coordinates $x$, $y$, and $z$, and $\nabla^2$ is the Laplace operator. The Laplace equation is named after Pierre-Simon Laplace, who first studied its properties. Solutions of Laplace's equation are called harmonic functions and are important in many areas of physics, including the study of electromagnetic fields, heat conduction and fluid dynamics. In two dimensions, Laplace's equation is given by

$$
\frac{\partial^2 V}{\partial x^2} + \frac{\partial^2 V}{\partial y^2} = 0
$$

whose solution can be iteratively found through Jacobi's method: if we discretize the domain in a grid of points, the value of each point can be updated as the average of its neighbors. The algorithm is as follows:

- initialize two matrices as in the following picture: the first matrix is filled with zeros, the second one with $0.5$, both with the same boundary conditions: $0$ in the upper and right boundaries, $100$ in the lower left corner, with increasing values starting from that corner and getting farther from it along the left and lower boundaries:
  
  ![init](imgs/init.png)
  
- Iterate over the grid points, updating each internal point of the first matrix as the average of its neighbors in the second matrix:

$$
V_{i,j}^{k+1} = \frac{1}{4} \left( V_{i-1,j}^k + V_{i+1,j}^k + V_{i,j-1}^k + V_{i,j+1}^k \right)
$$

- Swap the pointers of the two matrices and repeat points 2 and 3 until a desired convergence criterion is met.

The following gif shows the evolution of the matrix during 100 iterations:

![gif](imgs/solution.gif)

## Distribute the domain: MPI

Since at each iteration each point is updated independently on the others (we only need their old value, which is constant during the update), this algorithm clearly opens the door to parallelization: each process can be assigned a subgrid of the domain, and the communication between processes is only needed at the boundaries of the subgrids.

In this assignment, we will consider the domain to be distributed by rows among multiple MPI processes, hence each process will have a subgrid with a fixed number of rows of the entire grid (equal to the total number of rows divided by the number of processes), and **two more rows**, needed to perform the update, open for the other processes to access and update them through the use of two `MPI_Win` objects. Since in general the number of rows of the grid is not divisible by the number of processes, some processes will actually have one more row than the others:
![worksharing](imgs/worksharing.png)

![windows](imgs/windows.png)

For example, if `dim`$=9$ and `NPEs`$=3$, we have the situation showed in the following picture:

![sendrecgraph](imgs/sendrecgraph.png)

The idea to compute the solution is the following: each process has two submatrices with `myWorkSize = 9/3 + 0 = 3` rows, and 2 more rows to perform the update. Each process only initializes and updates one submatrix and then puts its first and last row inside the neighbor processes' windows. More precisely, each process first initializes its own submatrices and its extra rows, and then continuously:

- updates the values of the internal points of one submatrix (hence excluding its first and last row and the first and last column) using the values from the other one:
  
  ![update](imgs/update.png)

- updates the first and last row of the same submatrix, using the other one and the extra rows:
  
  ![updatebound](imgs/updatebound.png)

- puts the first row of the submatrix inside the upper process' second window and the last row of the submatrix inside the lower process' first one (first and last process only put a single row, since the other one is a fixed boundary condition):
  
  ![put](imgs/put.png)

- swaps the pointers to the matrices, so that the new matrix becomes the old one and vice versa;

until a desired convergence criterion is met.

## Results

In this section we will analyze the performances obtained by the algorithm. The code has been run on the Leonardo cluster, with up to 16 MPI tasks allocated one per node. The execution time has been measured with the `MPI_Wtime` function. The tests have been done with a matrix of size 1200x1200 and 12000x12000, with 10 evolution iterations, and 40000x40000, with 1000 iterations, to better study the scalability. The maximum time among all the MPI processes has been plotted. However, I have also collected data regarding the average time and they have showed the same behavior, meaning the workload is correctly distributed among the processes, for this reason they have not been plotted.

To easily identify the different parts of the code and plot them I have used some terms, here a brief explanation of them is given, in order of appearance in the code:
- `initPar`: parameters and windows initialization; 
- `init`: initialization of the matrices;
- `update`: total time spent on updating the matrix;
- `comm` total time spent on updating the extra rows;
- `save`: save the matrix on file using MPI-IO.

We'll also plot the results obtained with the standard Send/Recv communication, in order to compare the performances of the two methods (the first image will be the one-sided communication, the second one is the standard Send/Recv communication).

Let's start with the results obtained with the 1200x1200 matrix:

![cpu1200](imgs/results/120010.png) ![cpu1200](../Jacobi/imgs/results/cpu1200.png)

As we can see, there is no scalability due to the very low time spent: `initPar` takes more than half of the total time, and the time spent on `update` is negligible. Similar results were obtained with the standard Send/Recv communication, but in that case `init` was the only relevant part of the code.

Let's see how things change with a larger matrix:

![cpu12000](imgs/results/12k10.png) ![cpu12000](../Jacobi/imgs/results/cpu12000.png)

With a larger matrix we can start to appreciate some speedup, and the time spent on `update` is now significant, although `initPar` is still very relevant. We can observe as both the `init` and `update` parts of the code behave very similarly to the standard Send/Recv communication, but in that case the scalability is much better since there is no windows initialization.

Let's see what happens with a much larger matrix and more iterations:

![cpu40000](imgs/results/40k1000.png) ![cpu40000](../Jacobi/imgs/results/cpu40000.png)

We can finally appreciate a great scalability, with the time spent on `update` being the most relevant part of the code, as we would expect. `update` time is basically the same for both the one-sided and the standard Send/Recv communication, let's see how the other parts behave:

![cpu40000](imgs/results/40k1000noupd.png) ![cpu40000](../Jacobi/imgs/results/cpu40000noupd.png)

`init` still shows the same behavior in the two cases, while the communication time is far worse with the one-sided communication, especially with higher number of tasks.

### Save time

Up to now we have ignored the `save` time, let's now see how it behaves compared to the other parts of the code:

![save](imgs/results/1200save.png)

![save](imgs/results/12ksave.png)

As we can see, using MPI-IO we are able to save some time writing on file in parallel, but the time spent on this part is still by far the most time-consuming part of the code.

## How to run

A Makefile is provided to easily compile and run the code. The available targets are:

- `make`: produce an executable that prints the elapsed times; 
- `make save`: produce an executable that also saves the final matrix in a file `solution.dat`;
- `make gif`: produce an executable that also saves the evolution of the matrix in multiple `.dat` files;
- `make plot`: produce a plot using Gnuplot: if the code has been compiled with the `save` target, it will plot the final matrix in a file `solution.png`, while with the `gif` option it will plot a gif with the evolution of the matrix in a file `solution.gif`, both in the `output` folder;
- `make clean`: remove all the executables and the object files.

After compilation, the executables can be run with `mpirun -np <np> ./main <size> <nIter>`.

The Makefile also provides a shortcut to directly compile and run the code and save the output: `make run NP=<np> SZ=<size> IT=<nIter>`, equivalent to `make clean && make save && mpirun -np NP ./jacobi.x SZ IT && make plot`.

## Check correctness

In order to check correctness of the obtained output, the serial code is provided in [original_code](original_code/) folder, and a special target can be used to directly compare the output of the original code with the one of the optimized code: 
`make compare NP=<nProc> SZ=<size> IT=<nIter>`
This target will compile and run both the original and the optimized code (with the given number of processes, size and number of iterations), save the outputs in binary format, and compare them using Unix command `diff`: if the outputs are identical, as expected, no output will be produced, otherwise the output will be
```
Binary files output/solution0.dat and original_code/solution.dat differ
```

>**Side note**: MPI-IO writes binary files and does not truncate the file on which it'll write if it already exists: if you want to run the program with a size which is smaller than the previous one, delete the `solution.dat` file before running, in order to generate it from scratch instead of overwriting it. `compare` target is already provided with an internal `clean`, in order to repeatedly compare results without having to worry about non-truncated files.

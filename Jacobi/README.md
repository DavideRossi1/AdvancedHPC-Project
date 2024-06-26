# Exercise 2: Jacobi's Algorithm with OpenACC <!-- omit in toc -->

## Table of Contents <!-- omit in toc -->
- [Introduction](#introduction)
- [Jacobi's algorithm](#jacobis-algorithm)
- [Distribute the domain: MPI](#distribute-the-domain-mpi)
- [Move to GPU: OpenACC](#move-to-gpu-openacc)
- [Results](#results)
  - [CPU](#cpu)
  - [GPU](#gpu)
  - [Comparison](#comparison)
  - [Save time](#save-time)
- [How to run](#how-to-run)
- [Check correctness](#check-correctness)
  
## Introduction

The second assignment consists of implementing the Jacobi's method to solve Laplace equation in a distributed memory environment, using the MPI library to communicate between processes and OpenACC to parallelize the computation on GPU. The program is expected to run entirely on GPU, without any data transfer between CPU and GPU in the middle of the computation. 

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

In this assignment, we will consider the domain to be distributed by rows among multiple MPI processes, hence each process will have a subgrid with a fixed number of rows of the entire grid (equal to the total number of rows divided by the number of processes, plus two more rows, one above and one below, that will be needed to perform the update). Since in general the number of rows of the grid is not divisible by the number of processes, some processes will actually have one more row than the others:

![worksharing](imgs/worksharing.png)

For example, if `dim`$=9$ and `NPEs`$=3$, we have the situation showed in the following picture:

![sendrecgraph](imgs/sendrecgraph.png)

The idea to compute the solution is the following: each process will have two submatrices with `myWorkSize = 9/3 + 0 + 2 = 5` rows, also considering the 2 ghost rows, one above and one below, to perform the update. Each process only initializes and updates the internal rows of one submatrix, and then exchanges the boundary rows with the neighbor processes. More precisely, each process first initializes its own submatrices and then continuously:

- updates the values of the internal points (hence excluding its first and last row and the first and last column) of one submatrix using the values from the other one:
  
  ![update](imgs/update.png)

- updates the boundaries: it sends its second row to the upper process and its semilast row to the lower one, and receives its first row from the upper process and its last row from the lower one (first and last process only put a single row, since the other one is a fixed boundary condition):
  
  ![sendrec](imgs/sendrec.png)

- swaps the pointers to the matrices, so that the new matrix becomes the old one and vice versa;

until a desired convergence criterion is met.

## Move to GPU: OpenACC

The Jacobi's method is a perfect candidate for GPU acceleration, and OpenACC offers simple and powerful instruments to do so. The idea is to generate a `data` region to allocate the matrix on the GPU and perform both initialization and updates there, and then copy it back to the CPU:

![accdata](imgs/accdata.png)

Both initialization and update can then be parallelized using the `parallel loop` directive:

![initacc](imgs/initacc.png)

![updacc](imgs/updacc.png)

Also, in order to execute the the rows exchange directly between GPUs, `acc host_data use_device` directive has been used:

![sendrecacc](imgs/sendrecacc.png)

## Results

In this section we will analyze the performances obtained by the algorithm, both on CPU and on GPU. The code has been run on the Leonardo cluster, with up to 16 MPI tasks allocated one per node, for CPU versions, and up to 32 MPI tasks allocated four per node, one per GPU card, for the GPU version. The execution time has been measured with the `MPI_Wtime` function. The tests have been done with a matrix of size 1200x1200 and 12000x12000, with 10 evolution iterations, and 40000x40000, with 1000 iterations, to better study the scalability. The maximum time among all the MPI processes has been plotted. However, I have also collected data regarding the average time and they have showed the same behavior, meaning the workload is correctly distributed among the processes, for this reason they have not been plotted.

To easily identify the different parts of the code and plot them I have used some terms, here a brief explanation of them is given, in order of appearance in the code:
- `initacc`: initialization of OpenACC, with `acc_get_num_devices`, `acc_set_device_num` and `acc_init`;
- `copyin`: enter the data region, allocate the matrices on GPU;
- `init`: initialization of the matrices;
- `update`: total time spent on updating the matrix;
- `sendrecv` total time spent on exchanging the ghost rows;
- `save`: save the matrix on file using MPI-IO;
- `copyout`: copying the output matrix from GPU to CPU.

>**Note**: to further improve performances on CPU, OpenMP is used to parallelize both the initialization and the update of the matrices when GPUs are not available.

### CPU

Let's start with the CPU version:

![cpu1200](imgs/results/cpu1200.png)

As we can see, there is basically no scalability due to the very low time spent. `init` takes almost all the time, with `update` being practically irrelevant due to the very low number of updates done.

Let's see how things change with a larger matrix:

![cpu12000](imgs/results/cpu12000.png)

We can start to appreciate some speedup, and the time spent on `update` is now relevant, although `init` is still the most time-consuming part of the code, but scalability still almost interrupts after 4 tasks. 

Let's see what happens with a much larger matrix and more iterations:

![cpu40000](imgs/results/cpu40000.png)

We can finally appreciate a great scalability, with the time spent on `update` being the most relevant part of the code, as we would expect.

### GPU

Let's now move to the GPU version:

![gpu1200](imgs/results/gpu1200.png)

As we can see, it is totally pointless to run the code on GPU with such a small matrix, since most of the time is spent on `initacc`, hence we would get no speedup at all. Let's see what happens with a larger matrix:

![gpu12000](imgs/results/gpu12000.png)

We can start to appreciate some speedup, but the time spent on `initacc` is still relevant and most of the time is spent in `copyout` (with less tasks) or still in `initacc` (with more tasks), with `init` and `update` being basically negligible. This is due to the fact that the workload is too small to fully exploit the power of the GPU. Let's then try to run the code with a much larger matrix and many more iterations, in order to increase the workload:

![gpu40000](imgs/results/gpu40000.png)

We can finally appreciate a significant speedup even with a large number of MPI tasks: the time spent on `initacc` is now negligible with respect to the other parts of the code and most of the time is now spent on the update, as we would expect.

### Comparison

Let's now compare the CPU and GPU versions with the same matrix size and number of iterations:

![comp](imgs/results/comparison.png)

![comp](imgs/results/comparisonall.png)

As we can see, if with a small matrix the GPU version is not convenient at all, with a larger matrix we can appreciate a significant boost in performances. 

### Save time

Up to now we have ignored the `save` time, let's now see how it affects the performances: since it is not influenced by GPU acceleration, we'll just compare it with other parts of the code in order to understand its magnitude:

![save](imgs/results/1200save.png)

![save](imgs/results/12000save.png)

As we can see, using MPI-IO we are able to save some time writing on file in parallel, but this is still by far the most time-consuming part of the code.

## How to run

>*Disclaimer*: `%pu...` means that the target exists both as `cpu...` and `gpu...`

A Makefile is provided to easily compile and run the code. The available targets are:

- `make %pu`, `make %pusave` and `make %pugif`: produce an executable running on CPU with OpenMP or on GPU with OpenACC, the second one also saves the final matrix in a file `solution.dat`, while the third one saves the entire evolution of the matrix in multiple `.dat` files; 
- `make plot`: produce a plot using Gnuplot: if the code has been compiled with the `save` option, it will plot the final matrix in a file `solution.png`, while with the `gif` option it will plot a gif with the evolution of the matrix in a file `solution.gif`;
- `make clean`: remove all the executables and the object files.

After compilation, the executables can be run with `mpirun -np <np> ./jacobi.x <size> <nIter>`.

The Makefile also provides a shortcut to directly compile and run the code and save the output: `make %purun NP=<np> SZ=<size> IT=<nIter>`, equivalent to `make clean && make %pusave && mpirun -np NP ./jacobi.x SZ IT && make plot`;

## Check correctness

In order to check correctness of the obtained output, the original serial code is provided in [original_code](original_code/) folder, and a special target can be used to directly compare the output of the original code with the one of the optimized code: 
`make compare%pu NP=<nProc> SZ=<size> IT=<nIter>`
This target will compile and run both the original and the optimized code (with the given number of processes, size and number of iterations, on CPU or GPU), save the outputs in binary format, and compare them using Unix command `diff`: if the outputs are identical, as expected, no output will be produced, otherwise the output will be
```
Binary files output/solution0.dat and original_code/solution.dat differ
```
Note: to directly compare the two files without having to worry about precision issues, the original code `save_gnuplot` function has been modified to save binary files; this is the only change that has been performed on it.

>**Note**: MPI-IO writes binary files and does not truncate the file on which it'll write if it already exists: if you want to run the program with a size which is smaller than the previous one, `make clean` or empty the `output` folder before running, in order to generate it from scratch instead of overwriting it. `compare%pu` targets are already provided with an internal `clean`, in order to repeatedly compare results without having to worry about non-truncated files.

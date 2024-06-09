# Exercise 2: Jacobi's Algorithm <!-- omit in toc -->

## Table of Contents <!-- omit in toc -->
- [Introduction](#introduction)
- [Jacobi's algorithm](#jacobis-algorithm)
- [Distribute the domain: MPI](#distribute-the-domain-mpi)
- [Move to GPU: OpenACC](#move-to-gpu-openacc)
- [Results](#results)
- [How to run](#how-to-run)
  
## Introduction

The second assignment consists of implementing the Jacobi's method to solve Laplace equation in a distributed memory environment, using the MPI library to communicate between processes and OpenACC to parallelize the computation on GPU. The program is expected to run entirely on GPU, without any data transfer between CPU and GPU in the middle of the computation. 

Before digging into the implementation of the algorithm, let's first describe the problem and how to solve it.

## Jacobi's algorithm

Laplace's equation is a second-order partial differential equation, often written in the form

$$
\nabla^2 V = 0
$$

where $V$ is the unknown function of the spatial coordinates $x$, $y$, and $z$. The Laplace equation is named after Pierre-Simon Laplace, who first studied its properties. Solutions of Laplace's equation are called harmonic functions and are important in many areas of physics, including the study of electromagnetic fields, heat conduction and fluid dynamics. In two dimensions, Laplace's equation is given by

$$
\frac{\partial^2 V}{\partial x^2} + \frac{\partial^2 V}{\partial y^2} = 0
$$

whose solution can be iteratively found through Jacobi's method: if we discretize the domain in a grid of points, the value of each point can be updated as the average of its neighbors. The algorithm is as follows:

1. initialize the matrices as desired: the first matrix is filled with zeros, the second one with $0.5$, both with the same boundary conditions: $0$ in the upper and right boundaries, $100$ in the lower left corner, with increasing values starting from that corner and getting farther from it:

    ![init](imgs/init.png)

    This is done using 4 loops:
    - one to initialize both matrices with zeros;
    - one to set $0.5$ for the internal points of the second matrix;
    - one to set the first column;
    - one to set the last row;
2. Iterate over the grid points, updating each point as the average of its neighbors:

$$
V_{i,j}^{k+1} = \frac{1}{4} \left( V_{i-1,j}^k + V_{i+1,j}^k + V_{i,j-1}^k + V_{i,j+1}^k \right)
$$

3. Repeat step 2 until a desired convergence criterion is met.

## Distribute the domain: MPI

Since at each iteration each point is updated independently on the others, this algorithm clearly opens the door to parallelization: each process can be assigned a subgrid of the domain, and the communication between processes is only needed at the boundaries of the subgrids.

In this assignment, we will consider the domain to be distributed by rows among multiple MPI processes, hence each process will have a subgrid with a fixed number of rows of the entire grid (equal to the number of rows of the entire grid divided by the number of processes, plus two more rows, one above and one below, that will be needed to perform the update). Since in general the number of rows of the grid is not divisible by the number of processes, some processes will actually have one more row than the others:

![worksharing](imgs/worksharing.png)

For example, if `dim`$=9$ and `NPEs`$=3$, we have the situation showed in the following picture:

![worksharing](imgs/sendrecgraph.png)

each process will have a subgrid with 3 rows, plus 2 ghost rows, one above and one below, to perform the update. Each process will then send its semilast row, and receive its last row, to/from the upper process, and send its second row, and receive its first row, to/from the lower process. First (last) process will send and receive only one row, since its first (last) row is a fixed boundary condition.

The idea to compute the solution is the following: each process has two matrices, one for the current iteration and one for the next iteration, which are swapped at each iteration, and it:

- initalizes the two matrices as described above (each process initializes only its own subgrid, following the rules described above);

- iteratively updates the values of the new matrix using the old matrix: at each iteration:
  - updates the values of the internal points of the subgrid (hence excluding its first and last row and the first and last column, which are boundary conditions):
  
    ![update](imgs/update.png)

  - sends second and semilast row, and receives first and last row, to/from the neighboring processes, to update the boundary points, as described above:
  
    ![sendrec](imgs/sendrec.png)

  - swaps the pointers to the matrices, so that the new matrix becomes the old one and vice versa.
  
**Note**: to further improve performances on CPU, OpenMP has been used to parallelize both the initialization and the update of the matrices.

## Move to GPU: OpenACC

The Jacobi's method is a perfect candidate for GPU acceleration, and OpenACC offers simple and powerful instruments to do so. The main idea is to generate a `data` region to allocate the matrix on the GPU and perform both initialization and update there:

![accdata](imgs/accdata.png)

Both initialization and update can then be parallelized using the `parallel loop` directive:

![initacc](imgs/initacc.png)

![updacc](imgs/updacc.png)


## Results


## How to run

*Disclaimer*: `%pu...` means that target exists both as `cpu...` and `gpu...`

A Makefile is provided to easily compile and run the code. The available targets are:

- `make %pu`, `make %pusave` and `make %pugif`: produce an executable running on CPU with OpenMP or on GPU with OpenACC, the second one also saves the final matrix in a file `solution.dat`, while the third one saves the entire evolution of the matrix in multiple `.dat` files; 
- `make plot`: produce a plot using Gnuplot: if the code has been compiled with the `save` option, it will plot the final matrix in a file `solution.png`, while with the `gif` option it will plot a gif with the evolution of the matrix in a file `solution.gif`;
- `make clean`: remove all the executables and the object files.

After compilation, the executables can be run with `mpirun -np <np> ./jacobi.x <size> <nIter>`.

The Makefile also provides a shortcut to directly compile and run the code: `make %purun NP=<np> SZ=<size> IT=<nIter>`, equivalent to `make clean && make %pusave && mpirun -np NP ./jacobi.x SZ IT`;

In order to check correctness of the obtained output, the original code is provided in [original_code](original_code/) folder, and a special target can be used to directly compare the output of the original code with the one of the optimized code: 
`make compare%pu NP=<nProc> SZ=<size> IT=<nIter>`
This target will compile and run both the original and the optimized code (with the given number of processes, size and number of iterations, on CPU or GPU), produce the output images in `.png` format, and compare them using Unix command `diff`: if the images are identical, as expected, no output will be produced, otherwise the output will be
```
Binary files solution.png and original_code/solution.png differ
```


**Side note**: MPI-IO has been used to save the matrix on file. The file is saved in binary format, and this causes some issues if the file is overwritten multiple times, since MPI-IO does not truncate the file before writing: if you want to run the program with a size which is smaller than the previous one, delete the `solution.dat` file before running, in order to generate it from scratch instead of overwriting it.




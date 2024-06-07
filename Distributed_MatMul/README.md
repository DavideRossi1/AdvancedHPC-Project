# Exercise 1: Distributed Matrix-Matrix Multiplication

## Table of Contents

- [Exercise 1: Distributed Matrix-Matrix Multiplication](#exercise-1-distributed-matrix-matrix-multiplication)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Matrix-Matrix Multiplication using MPI](#matrix-matrix-multiplication-using-mpi)
  - [Basic version](#basic-version)
  - [Improved CPU version](#improved-cpu-version)
  - [GPU version](#gpu-version)
  - [Results](#results)
  - [How to run](#how-to-run)
  
## Introduction

The first assignment consists of implementing a distributed matrix-matrix multiplication, using the MPI library to communicate between processes. More precisely, 3 versions of the algorithm are required:
- a basic version with the naive algorithm (triple loop);
- an improved CPU version using BLAS library;
- a GPU version using CUDA and CUBLAS library.

Before digging into the implementation of the three versions, let's first describe the problem and how to solve it.

## Matrix-Matrix Multiplication using MPI

Matrix-matrix multiplication is a fundamental operation in linear algebra and a good exercise to implement in a distributed environment, and consists in computing $C=A\times B$, where A is a $m\times n$ matrix, B is a $n\times l$ matrix and the output $C$ is a $m\times l$ matrix. The implementation of a distributed matrix-matrix multiplication lies on two main concepts:
- matrices are saved by rows in contiguous memory;
- each of the three matrices is distributed among the processes.

For this assignment, we will consider the matrices to be distributed by rows among the processes, hence each process will have a submatrix, which we will call `myA`, `myB` and `myC`, with a fixed number of rows of each matrix (equal to the number of rows of the entire matrix divided by the number of processes). Since in general the number of rows of the matrices is not divisible by the number of processes, some processes will actually have one more row than the others:

![worksharing](imgs/workshare.png)

The idea to compute the product is the following: iterate over the number of processes: at each iteration, each process:
- re-builds a group of columns of $B$, named `columnB`, by gathering the necessary part from all the other processes;
- computes `myCBlock = myA × columnB`;
- places `myCBlock` in `myC`: the union of the `myCBlock`s of the current iteration will give a group of columns of the final matrix $C$.  
 
Essentially, $C$ matrix is built by columns: at iteration `i+1`, for `i=0,...NPEs-1`, each process computes its `myNRows` rows of a block of $k$ columns, where $k$ is the worksize of the `i`-th process.

![matrix-matrix](imgs/mult.png)

The product above is computed in 3 iterations, as:

![it1](imgs/it1.png)

![it2](imgs/it2.png)

![it3](imgs/it3.png)

where `columnB` is made by the current process part, in yellow, and the parts sent by the other two processes, in blue and pink, and `myCBlock`, in green, is computed and placed in the correct position in `myC`. Note that no process will ever store any of the matrices in their entirety, but only the part they need to compute their part of the product.


The code that executes the iterations is:

![MMcode](imgs/MM_code.png)

What the `matMul` function does depends on the version of the algorithm we are implementing. We'll now see some details about the three versions.

## Basic version

The basic version of the algorithm is the naive implementation of the matrix-matrix multiplication, using the triple loop:

![basic](imgs/basic.png)

`startingCol` is a shift that allows to directly position `myCBlock` in `myC`. Except for this, the code is straightforward: each process computes its part of `myC` by iterating over the rows of `myA` and the columns of `columnB`.

## Improved CPU version

The improved CPU version uses the BLAS library to compute the matrix-matrix multiplication. The BLAS library is a set of routines that provide standard building blocks for performing basic vector and matrix operations. The routine we are interested in is `dgemm`, which computes the matrix-matrix product of two matrices with double-precision elements. The code here is just a little bit more complex than the basic version: product and `myCBlock` placement are split in two different steps:

![blas](imgs/blas.png)

Notice that, to avoid losing performances, we are specifying to `dgemm` that we don't want to transpose the matrices.

## GPU version

GPU execution, which is done with CUDA, requires some more steps with respect to the previous two versions:

![cuda](imgs/cuda.png)

- allocate memory for the matrices on GPU: `myA` has already been allocated before the main loop since it is fixed, but `columnB` and consequently `myCBlock` change size at each iteration, hence we need to allocate memory for them every time;
- copy `columnB` from CPU to GPU, and allocate memory on CPU for `myCBlock`;
- execute the product (this time on GPU using the `cublasDgemm` function) and copy the result back to CPU;
- place `myCBlock` in `myC`.

## Results


## How to run

A Makefile is provided to easily compile and run the code. The available targets are:

- `make naive`: produce an executable running with the naive algorithm (triple loop);
- `make cpu`: produce an executable running with the BLAS library; 
- `make gpu`: produce an executable running with CUDA and CUBLAS library;
- `make clean`: remove all the executables and the object files.

After compilation, the executables can be run with `mpirun -np <np> ./main <size>`.

The Makefile also provides some shortcuts to directly compile and run the code:

- `make naiverun NP=<np> SZ=<size>`: equivalent to `make clean && make naive && mpirun -np <np> ./main <size>`;
- `make cpurun NP=<np> SZ=<size>`: equivalent to `make clean && make cpu && mpirun -np <np> ./main <size>`;
- `make gpurun NP=<np> SZ=<size>`: equivalent to `make clean && make gpu && mpirun -np <np> ./main <size>`.

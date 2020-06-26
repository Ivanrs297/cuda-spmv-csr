# CSR Sparse Matrix-Vector Multiplication

Sparse matrix-vector multiplication (SpMV) is a computational kernel of the form y = Ax. The matrix A is a sparse matrix, that is a matrix in which most of the elements are zero.

The CSR (Compressed Sparse Row) is an array representation of a matrix which consist on three one-dimensional arrays called as values, index, row ptr. The values array is of size NNZ (Non-zero values) and stores the values of the non-zero elements of the matrix. Then, the index array stores the column index of each element in the values array. Finally, the row ptr array is of size m + 1 and stores the cumulative number of non-zero elements upto the ith row.

## Algorithm:

1. Use row_ptr array to create the segmented values array.
2. Use value[n] ∗ X[index[n]] map operation.
3. Execute the segmented inclusive scan on each row of
   the matrix.
4. Use row ptr to collect the output of each block into the Y dense vector

## Use and Installation

Requires CUDA enviroment.

Test datasets can be found on /data/ folder

Compile:

```sh
$ cd cuda-spmv-csr
$ nvcc sparse_matrix.cu -o sparse_matrix -lm
```

Run:

```sh
$ ./sparse_matrix NNZ ROWS COLS DEBUG
```

#### Inputs:

- NNZ: None Zero Values
- ROWS: The number of Rows (max 1024)
- COLS: The number of Columns (max 1024)
- DEBUG: 1 to debug, 0 to no-debug

#### Output:

- Error in Y (if DEBUG is active)
- Time in MS
- Throughput in GFLOPS

## Todos

- Develop inclusive segmented scan across several blocks

## License

MIT

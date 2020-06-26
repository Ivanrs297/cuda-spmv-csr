/*

sparse_matrix.cu:
	Cuda implementation Sparse Matrix Multiplication by Vector

compile & run:
	nvcc sparse_matrix.cu -o sparse_matrix -lm && ./sparse_matrix.sh 32768 256 256 1


input:
	NNZ: None Zero Values
	ROWS: The number of Rows (max 1024)
	COLS: The number of Columns (max 1024)
	DEBUG: 1 to debug, 0 to no-debug

output:
	Time in MS
	Throughput in GFLOPS

author:     Ivan Reyes-Amezcua
date:       June, 2020

*/

#include <stdio.h>
#include <math.h>
#include <time.h> 
#include <iostream>
#include <fstream>
#include <stdlib.h>

using namespace std;

__global__ void spmv(int num_rows, int num_cols, int *row_ptrs,
	int *col_index, double *values, double *x, double *y
) {
	extern __shared__ double sum[];  // sum of the values per row per block
	int tid =  threadIdx.x;  // Local Thread ID
	int start_row_ptr = row_ptrs[blockIdx.x];  // Starting index of the row per block
	int finish_row_ptr = row_ptrs[blockIdx.x + 1];  // Finishing index of the row per block
	sum[threadIdx.x] = 0.0; 
	__syncthreads();

	if (tid + start_row_ptr >= start_row_ptr &&
		tid + start_row_ptr < finish_row_ptr)
	{
		
		// TODO: check col_index vector, possible memory issue
		sum[tid] = values[tid + start_row_ptr] * x[col_index[tid + start_row_ptr]]; // Map: value[n] * X[index[n]]
	}
	__syncthreads();
	
	// Inclusive Scan
	for (int j = 0; j < (int)log2((float) blockDim.x); j++ ){
		if ( tid - pow(2, j) >= 0) {
			sum[tid] += sum[tid - (int)pow(2, j)];
		}
		__syncthreads();
	}

	// Save the result of Row-Block on global memory
    if(tid == blockDim.x - 1){
		y[blockIdx.x] = sum[tid];
	}
}

int main(int argc, char *argv[]) {

	// Get and validate arguments
    if(argc != 5){
        printf("Usage %s NNZ ROWS COLS DEBUG\n",argv[0]);
        exit(0);
	}
	
	int NNZ = atoi ( argv[1] );  		// Non Zero Values
	int num_rows = atoi ( argv[2] );	// rows
	int num_cols = atoi ( argv[3] );	// columns
	int debug = atoi ( argv[4] );  		// 1 for debug, 0 for NO-debug

	double values[NNZ];  				// CSR format
	int col_index[NNZ]; 				// CSR format
	int row_ptrs[num_rows + 1];  		// CSR format
	double x[num_cols]; 				// the vector to multiply
	double y[num_rows];  				// the output
	double true_y[num_rows];  			// the true Y results of operation

	// Declare GPU memory pointers
	double *d_values;
	double *d_x;
	double *d_y;
	int *d_col_index;
	int *d_row_ptrs;

	// Allocate GPU memory
	int r1 = cudaMalloc((void **) &d_values, NNZ*sizeof( double ));
	int r2 = cudaMalloc((void **) &d_x, num_cols*sizeof( double ));
	int r3 = cudaMalloc((void **) &d_y, num_rows*sizeof( double ));
	int r4 = cudaMalloc((void **) &d_col_index, NNZ*sizeof( int ));
	int r5 = cudaMalloc((void **) &d_row_ptrs, (num_rows + 1)*sizeof( int ));
	if( r1 || r2 || r3 || r4 || r5 ) {
	   printf( "Error allocating memory in GPU\n" );
	   exit( 0 );
	}

	// Read the Values and Index:
	std::ifstream values_file("./data/values.txt");
	std::ifstream col_ind_file("./data/col_ind.txt");
    for (int i = 0; i < NNZ; i++) {
		values_file >> values[i];

		double aux;
		col_ind_file >> aux;
		col_index[i] = (int) aux;
	}

	// Read the row_ptr and the True Ys:
	std::ifstream row_ptr_file("./data/row_ptr.txt");
	std::ifstream true_y_file("./data/y.txt");
    for (int i = 0; i < (num_rows + 1); i++) {
		double aux, aux2;

		row_ptr_file >> aux;
		true_y_file >> aux2;

		row_ptrs[i] = (int) aux;
		true_y[i] = (int) aux2;
	}
	
	// Read the X values:
	std::ifstream x_file("./data/x.txt");
	for (int i = 0; i < num_cols; i++) {
		x_file >> x[i];
	}

	// Transfer the arrays to the GPU:
	cudaMemcpy(d_values, values, NNZ*sizeof( double ), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, x, num_cols*sizeof( double ), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, num_rows*sizeof( double ), cudaMemcpyHostToDevice);
	cudaMemcpy(d_col_index, col_index, NNZ*sizeof( int ), cudaMemcpyHostToDevice);
	cudaMemcpy(d_row_ptrs, row_ptrs, (num_rows + 1)*sizeof( int ), cudaMemcpyHostToDevice);

	// Start Time:
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	
	// Call to kernel:
	double size_sharedmem = num_cols*sizeof(double);  // Size of shared memory
	spmv<<<num_rows, num_cols, size_sharedmem>>>(num_rows, num_cols, d_row_ptrs, d_col_index, d_values, d_x, d_y);
	cudaDeviceSynchronize();

	// Stop Time:
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	// Transfer the values to the CPU:
	cudaMemcpy(y, d_y, num_rows * sizeof(double), cudaMemcpyDeviceToHost);

	// Get the error:
	int errors = 0;   // count of errors
	float e = 500.0;  // tolerance to error
	for (int i = 0; i < num_rows; i++) {
		if (abs(true_y[i] - y[i]) > e) {
			errors++;
			if(debug == 1)
				printf("Error in Y%d, True: %f, Calc: %f\n", i, true_y[i], y[i]);
		}
	}
	float error_rate = ((double)errors/(double)num_rows) * 100.0;
	float density = ((float)NNZ/((float)num_cols*(float)num_rows))*100.0;
	printf("\nM. Density: %0.2f%%, #Ys: %d, Errors: %d, Error Rate: %0.2f%%\n", density, num_rows, errors, error_rate);

	// Free Memory
	cudaFree( d_values );
	cudaFree( d_x );
	cudaFree( d_y );
	cudaFree( d_col_index );
	cudaFree( d_row_ptrs );

	// Calculate Throughput:
	float bw;
	bw = (float )num_rows*(float )num_cols*log2((float) num_cols);
	bw /= milliseconds*1000000.0;
	printf( "\nSpmV GPU execution time: %7.3f ms, Throughput: %6.2f GFLOPS\n\n", milliseconds, bw ); 

	// Store Runtime
	FILE *pFile = fopen("GPU_results.txt","a");
    fprintf(pFile, "%d, %0.2f, %0.2f, %d, %d, %7.3f, %6.2f\n", NNZ, density, error_rate, num_cols, num_rows, milliseconds, bw);
	fclose(pFile);
	

    return 0;
}
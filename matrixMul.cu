#include <stdio.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
//Code written by Alan Fleming

//CONSTANTS
#define MATRIXSIZE 8
#define BLOCKSIZE 4

void mul_matrix_cpu(float *M, float *N, float *P, int width){
	for( int i = 0; i<width; i++){
		for( int j = 0; j<width; j++){
			float sum = 0;
			for (int k = 0; k < width; k++){
				sum += M[i * width + k] * N[k * width + j];
			}
			P[i * width + j] = sum;
		}
	}
}

__global__ void mul_matrix_gpu(float *M, float *N, float *P, int width) {
	//Assuming matrix is width x width
	//Assuming tile size = blockdim.x
	__shared__ float ds_M[MATRIXSIZE * MATRIXSIZE];
	__shared__ float ds_N[MATRIXSIZE * MATRIXSIZE];

	//Calculate row and collumn
	int row = blockIdx.y * blockDim.x + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	//initialize Pvalue
	float Pvalue = 0;

	for (int i = 0; i < (width / blockDim.x); ++i) {
		//copy global memory into shared memory
		ds_M[row  * width + col] = M[row * width + i * blockDim.x + threadIdx.x];
		ds_N[row * width + col] = N[col + (i * blockDim.x + threadIdx.y) * width];		
		//ensure all data is copied
		__syncthreads();
		
		//Preform partial multiplications
		for(int k = 0; k < blockDim.x; ++k) {
			Pvalue += ds_M[threadIdx.y * width + k] * ds_N[k * width + threadIdx.x];
		}
		__syncthreads();
	}
	//Load final product into output memory
	P[row * width + col] = Pvalue;
}

bool verify(float *A, float *B, float *C, int width) {
	//Tolerance to check
	const float tolerance = 1e-6;
	for(int i = 0; i < width; ++i){
		for(int k = 0; k < width; ++k) {
			float sum = 0;
			for(int j = 0; j < width; ++j) {
				sum += A[i * width + j] * B[j * width + k];
			}
			
			//get the absolute value of the error for comparison
			float error = fabs(sum - C[i * width + k])/sum;
			//Check if error is too large
			if(error > tolerance) {
				printf("TEST FAILED\n\n");
				return false;
			}
		}
	}
	printf("TEST PASSED\n\n");
	return true;
}

int main(int argc, char *argv[]){
	//allocate system memory for array
	float *a = (float *)malloc(sizeof(float) * MATRIXSIZE * MATRIXSIZE );	//first matrix
	float *b = (float *)malloc(sizeof(float) * MATRIXSIZE * MATRIXSIZE ); //second matrix
	float *c = (float *)malloc(sizeof(float) * MATRIXSIZE * MATRIXSIZE ); //resulting matrix

	int init =1325;
	for (int i=0;i<MATRIXSIZE;i++){
	    for (int j=0;j<MATRIXSIZE;j++){
		init= 3125 * init % 6553;
		a[i * MATRIXSIZE + j]= ( init -1000 ) % 6553;
		b[i * MATRIXSIZE + j]= init % 251;
	    }
	}
	
	//get cpu start time
	clock_t t1 = clock();
	//run function
	mul_matrix_cpu(a, b, c, MATRIXSIZE);
	//get cpu stop time
	clock_t t2 = clock();
	//calculate runtime
	float cpuTime = (float(t2 - t1)/CLOCKS_PER_SEC*1000);

	//allocate memory on gpu
	float *dev_a, *dev_b, *dev_c;
	cudaMalloc((void **)(&dev_a),MATRIXSIZE * MATRIXSIZE * sizeof(float));
	cudaMalloc((void **)(&dev_b),MATRIXSIZE * MATRIXSIZE * sizeof(float));
	cudaMalloc((void **)(&dev_c),MATRIXSIZE * MATRIXSIZE * sizeof(float));

	//copy matrices to gpu
	cudaMemcpy(dev_a,a, MATRIXSIZE * MATRIXSIZE * sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b,b, MATRIXSIZE * MATRIXSIZE * sizeof(float),cudaMemcpyHostToDevice);

	//calculate dimensions for gpu
	dim3 dimBlock(BLOCKSIZE,BLOCKSIZE);
	dim3 dimGrid( ceil(double(MATRIXSIZE)/dimBlock.x), ceil(double(MATRIXSIZE) /dimBlock.y));

	//Set up cuda events for recording runtime
	cudaEvent_t start,stop;
	float gpuTime; 
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);

	// do some work on the GPU
	mul_matrix_gpu<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c, MATRIXSIZE);

	//calculate runtime 
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpuTime,start,stop);

	//destroy cuda events
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	//copy memory from device
	cudaMemcpy(c,dev_c, MATRIXSIZE * MATRIXSIZE * sizeof(int),cudaMemcpyDeviceToHost);

	//print results
	printf("CPU Runtime: %f\nGpu Runtime: %f\nSpeedup: %f\n", (double)cpuTime, (double)gpuTime, double(gpuTime / cpuTime));

	//verify results
	verify(a,b,c, MATRIXSIZE);
}

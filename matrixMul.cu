#include <stdio.h>
#include <stdio.h>
//Code written by Alan Fleming

void mul_matrix_cpu(float *M, float *N, float *P, int width){
	for( int i = 0; i<width; i++){
		for( int j = 0; j<width; j++){
			float sum = 0;
			for (float k = 0; k < width; k++){
				sum += M[i * width + k] * N[k * width + j];
			}
			P[i * width + j] = sum;
		}
	}
}

__global__ void mul_matrix_gpu(float *M, float *N, float *P, int width) {
	//Assuming matrix is width x width
	//Assuming tile size = blockdim.x
	__shared__ float ds_M[blockDim.x, blockDim.x];
	__shared__ float ds_N[blockDim.x, blockDim.x];

	//Calculate row and collumn
	int row = blockIdx.y * blockDim.x + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	//initialize Pvalue
	float Pvalue = 0;

	for (int i = 0; i < (width / blockDim.x); ++i) {
		//copy global memory into shared memory
		ds_M[threadIdx.y][threadIdx.x] = M[row * width + i * blockDim.x + threadIdx.x];
		ds_N[threadIdx.y][threadIdx.x] = N[col + (i * blockDim.x + threadIdx.y) * width];		
		//ensure all data is copied
		__synchthreads();
		
		//Preform partial multiplications
		for(int k = 0; k < blockDim.x; ++k) {
			Pvalue += ds_M[threadIdx.y][k] * ds_N[k][threadIdx.x];
		}
		__synchthreads();
	}
	//Load final product into output memory
	P[row * width + col] = Pvalue;
}



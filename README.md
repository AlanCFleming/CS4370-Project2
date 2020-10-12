# CS4370-Project2
This is a cuda program that covers "Tiled Matrix Multiplication" for class.

## Editing BLOCKSIZE and MATRIXSIZE
* A define statement for BLOCKSIZE can be found on line 15 of the .cu file
* A define statement for MATRIXSIZE can be found on line 14 of the .cu file

## Compiling
nvcc was used to compile these programs. This will create an executable program.
* Command for compiling matrix multiplication: nvcc Fleming-MatrixMul.cu -o MatrixMul

## Running
These programs can be run directly from the command line. 
* Command for matrix Multiplication: {path}/MatrixMul

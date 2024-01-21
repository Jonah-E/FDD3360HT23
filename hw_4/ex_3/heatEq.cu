#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cusparse_v2.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>

#define gpuCheck(stmt)                                                         \
  do {                                                                         \
    cudaError_t err = stmt;                                                    \
    if (err != cudaSuccess) {                                                  \
      printf("ERROR. Failed to run stmt %s\n", #stmt);                         \
      break;                                                                   \
    }                                                                          \
  } while (0)

// Macro to check the cuBLAS status
#define cublasCheck(stmt)                                                      \
  do {                                                                         \
    cublasStatus_t err = stmt;                                                 \
    if (err != CUBLAS_STATUS_SUCCESS) {                                        \
      printf("ERROR. Failed to run cuBLAS stmt %s\n", #stmt);                  \
      break;                                                                   \
    }                                                                          \
  } while (0)

// Macro to check the cuSPARSE status
#define cusparseCheck(stmt)                                                    \
  do {                                                                         \
    cusparseStatus_t err = stmt;                                               \
    if (err != CUSPARSE_STATUS_SUCCESS) {                                      \
      printf("ERROR. Failed to run cuSPARSE stmt %s\n", #stmt);                \
      break;                                                                   \
    }                                                                          \
  } while (0)

struct timeval t_start, t_end;
void cputimer_start() { gettimeofday(&t_start, 0); }
void cputimer_stop(const char* info)
{
  gettimeofday(&t_end, 0);
  double time = (1000000.0 * (t_end.tv_sec - t_start.tv_sec) + t_end.tv_usec -
                 t_start.tv_usec);
  printf("Timing - %s. \t\tElasped %.0f microseconds \n", info, time);
}

// Initialize the sparse matrix needed for the heat time step
void matrixInit(double* A, int* ArowPtr, int* AcolIndx, int dimX, double alpha)
{
  // Stencil from the finete difference discretization of the equation
  double stencil[] = {1, -2, 1};
  // Variable holding the position to insert a new element
  size_t ptr = 0;
  // Insert a row of zeros at the beginning of the matrix
  ArowPtr[1] = ptr;
  // Fill the non zero entries of the matrix
  for (int i = 1; i < (dimX - 1); ++i) {
    // Insert the elements: A[i][i-1], A[i][i], A[i][i+1]
    for (int k = 0; k < 3; ++k) {
      // Set the value for A[i][i+k-1]
      A[ptr] = stencil[k];
      // Set the column index for A[i][i+k-1]
      AcolIndx[ptr++] = i + k - 1;
    }
    // Set the number of newly added elements
    ArowPtr[i + 1] = ptr;
  }
  // Insert a row of zeros at the end of the matrix
  ArowPtr[dimX] = ptr;
}

int main(int argc, char** argv)
{
  int device = 0;              // Device to be used
  int dimX;                    // Dimension of the metal rod
  int nsteps;                  // Number of time steps to perform
  double alpha = 0.4;          // Diffusion coefficient
  double* temp;                // Array to store the final time step
  double* A;                   // Sparse matrix A values in the CSR format
  int* ARowPtr;                // Sparse matrix A row pointers in the CSR format
  int* AColIndx;               // Sparse matrix A col values in the CSR format
  int nzv;                     // Number of non zero values in the sparse matrix
  double* tmp;                 // Temporal array of dimX for computations
  size_t bufferSize = 0;       // Buffer size needed by some routines
  void* buffer = nullptr;      // Buffer used by some routines in the libraries
  int concurrentAccessQ;       // Check if concurrent access flag is set
  double zero = 0;             // Zero constant
  double one = 1;              // One constant
  double norm;                 // Variable for norm values
  double error;                // Variable for storing the relative error
  double tempLeft = 200.;      // Left heat source applied to the rod
  double tempRight = 300.;     // Right heat source applied to the rod
  cublasHandle_t cublasHandle; // cuBLAS handle
  cusparseHandle_t cusparseHandle;  // cuSPARSE handle
  cusparseSpMatDescr_t Adescriptor; // Mat descriptor needed by cuSPARSE
  cusparseDnVecDescr_t tempDesc;
  cusparseDnVecDescr_t tmpDesc;

  // Read the arguments from the command line
  dimX = atoi(argv[1]);
  nsteps = atoi(argv[2]);

  // Print input arguments
  printf("The X dimension of the grid is %d \n", dimX);
  printf("The number of time steps to perform is %d \n", nsteps);

  // Get if the cudaDevAttrConcurrentManagedAccess flag is set
  gpuCheck(cudaDeviceGetAttribute(&concurrentAccessQ,
                                  cudaDevAttrConcurrentManagedAccess, device));

  // Calculate the number of non zero values in the sparse matrix. This number
  // is known from the structure of the sparse matrix
  nzv = 3 * dimX - 6;

  cputimer_start();
  gpuCheck(cudaMallocManaged(&temp, sizeof(double) * dimX));
  gpuCheck(cudaMallocManaged(&A, sizeof(double) * nzv));
  gpuCheck(cudaMallocManaged(&ARowPtr, sizeof(double) * (dimX + 1)));
  gpuCheck(cudaMallocManaged(&AColIndx, sizeof(double) * nzv));
  gpuCheck(cudaMallocManaged(&tmp, sizeof(double) * dimX));
  cputimer_stop("Allocating device memory");

  // Check if concurrentAccessQ is non zero in order to prefetch memory
  if (concurrentAccessQ) {
    cputimer_start();
    cudaGetDevice(&device);
    cudaMemPrefetchAsync(temp, sizeof(double) * dimX, cudaCpuDeviceId, NULL);
    cudaMemPrefetchAsync(A, sizeof(double) * nzv, cudaCpuDeviceId, NULL);
    cudaMemPrefetchAsync(ARowPtr, sizeof(double) * (dimX + 1), cudaCpuDeviceId,
                         NULL);
    cudaMemPrefetchAsync(AColIndx, sizeof(double) * nzv, cudaCpuDeviceId, NULL);
    cudaMemPrefetchAsync(tmp, sizeof(double) * dimX, cudaCpuDeviceId, NULL);
    cudaDeviceSynchronize();
    cputimer_stop("Prefetching GPU memory to the host");
  }

  // Initialize the sparse matrix
  cputimer_start();
  matrixInit(A, ARowPtr, AColIndx, dimX, alpha);
  cputimer_stop("Initializing the sparse matrix on the host");

  // Initiliaze the boundary conditions for the heat equation
  cputimer_start();
  memset(temp, 0, sizeof(double) * dimX);
  temp[0] = tempLeft;
  temp[dimX - 1] = tempRight;
  cputimer_stop("Initializing memory on the host");

  if (concurrentAccessQ) {
    cputimer_start();
    cudaGetDevice(&device);
    cudaMemPrefetchAsync(temp, sizeof(double) * dimX, device, NULL);
    cudaMemPrefetchAsync(A, sizeof(double) * nzv, device, NULL);
    cudaMemPrefetchAsync(ARowPtr, sizeof(double) * (dimX + 1), device, NULL);
    cudaMemPrefetchAsync(AColIndx, sizeof(double) * nzv, device, NULL);
    cudaMemPrefetchAsync(tmp, sizeof(double) * dimX, device, NULL);
    cudaDeviceSynchronize();
    cputimer_stop("Prefetching GPU memory to the device");
  }

  cublasCheck(cublasCreate(&cublasHandle));

  cusparseCheck(cusparseCreate(&cusparseHandle));

  cublasCheck(cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_HOST));
  cusparseCheck(
      cusparseSetPointerMode(cusparseHandle, CUSPARSE_POINTER_MODE_HOST));

  cusparseCheck(cusparseCreateCsr(&Adescriptor, dimX, dimX, nzv,
                                  (void*) ARowPtr, (void*) AColIndx, (void*) A,
                                  CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                  CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
  cusparseCheck(cusparseCreateDnVec(&tempDesc, dimX, (void*) temp, CUDA_R_64F));
  cusparseCheck(cusparseCreateDnVec(&tmpDesc, dimX, (void*) tmp, CUDA_R_64F));

  cusparseCheck(cusparseSpMV_bufferSize(
      cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, Adescriptor,
      tempDesc, &zero, tmpDesc, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
      &bufferSize));
  gpuCheck(cudaMalloc(&buffer, bufferSize));

  // Perform the time step iterations
  for (int it = 0; it < nsteps; ++it) {
    cusparseCheck(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &one, Adescriptor, tempDesc, &zero, tmpDesc,
                               CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buffer));
    cudaDeviceSynchronize();

    cublasCheck(cublasDaxpy(cublasHandle, dimX, &alpha, tmp, 1, temp, 1));
    cudaDeviceSynchronize();

    cublasCheck(cublasDnrm2(cublasHandle, dimX, tmp, 1, &norm));
    cudaDeviceSynchronize();

    // If the norm of A*temp is smaller than 10^-4 exit the loop
    if (norm < 1e-4)
      break;
  }

  // Calculate the exact solution using thrust
  thrust::device_ptr<double> thrustPtr(tmp);
  thrust::sequence(thrustPtr, thrustPtr + dimX, tempLeft,
                   (tempRight - tempLeft) / (dimX - 1));

  // Calculate the relative approximation error:
  cublasCheck(cublasDnrm2(cublasHandle, dimX, tmp, 1, &norm));
  cudaDeviceSynchronize();

  one = -1;
  cublasCheck(cublasDaxpy(cublasHandle, dimX, &one, temp, 1, tmp, 1));
  cudaDeviceSynchronize();

  cublasCheck(cublasDnrm2(cublasHandle, dimX, tmp, 1, &norm));
  cudaDeviceSynchronize();
  error = norm;

  cublasCheck(cublasDnrm2(cublasHandle, dimX, temp, 1, &norm));
  cudaDeviceSynchronize();

  // Calculate the relative error
  error = error / norm;
  printf("The relative error of the approximation is %f\n", error);

  cusparseCheck(cusparseDestroyDnVec(tempDesc));
  cusparseCheck(cusparseDestroyDnVec(tmpDesc));
  cusparseCheck(cusparseDestroySpMat(Adescriptor));

  cusparseCheck(cusparseDestroy(cusparseHandle));

  cublasCheck(cublasDestroy(cublasHandle));

  cudaFree(temp);
  cudaFree(A);
  cudaFree(ARowPtr);
  cudaFree(AColIndx);
  cudaFree(tmp);
  cudaFree(buffer);

  return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#ifdef USE_FLOAT
#define DataType float
#pragma message ( "Compiling with DataType float" )
#else
#define DataType double
#endif

#define TPW 32
#define BLOCK_X_SIZE TPW
#define BLOCK_Y_SIZE TPW

/* Support function to generate a random number in a given range.
 * Based on the solution presented in the following forum thread:
 * http://ubuntuforums.org/showthread.php?t=1717717&p=10618266#post10618266*/
static DataType randfrom(DataType min, DataType max) {
  DataType range = (max - min);
  DataType div = RAND_MAX / range;
  return min + (rand() / div);
}

/* Populate a given matrix memory location with values from a given range.*/
static void generateRandMatrix(DataType *matrix, int rows, int columns, DataType min,
                        DataType max) {
  srand(time(NULL));

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < columns; ++j) {
      matrix[(i * columns) + j] = randfrom(min, max);
    }
  }
}

/*Get the current CPU time in seconds as a double.*/
static double getCpuSeconds(void) {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

/* Function to calculate the Euclidian norm on the difference between two matrices.*/
static DataType euclidianNormTwoMatrices(DataType *matrixA, DataType *matrixB,
                                  int rows, int columns) {
  DataType diffEu = 0;
  DataType *result = (DataType *)malloc(sizeof(DataType) * rows * columns);

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < columns; ++j) {
      result[i * columns + j] =
          matrixA[i * columns + j] - matrixB[i * columns + j];
      diffEu += result[i * columns + j] * result[i * columns + j];
    }
  }
  diffEu = sqrt(diffEu);

  free(result);
  return diffEu;
}

/* CUDA Kernel to perform the GEMM, designed to launch one thread per element in the C
 * output matrix.*/
__global__ void gemmGPU2D(DataType *A, DataType *B, DataType *C, int numARows,
                          int numAColumns, int numBRows, int numBColumns) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;

  if (idx < numBColumns && idy < numARows) {
    DataType result = 0.0;

    /* Do the calculation for the element idx, idy in the output matrix C.*/
    for (int k = 0; k < numAColumns; ++k) {
      result += A[idy * numAColumns + k] * B[k * numBColumns + idx];
    }
    C[idy * numBColumns + idx] = result;
  }
}

/* CPU function to perform the GEMM, C = A x B */
static void gemmCPU(DataType *A, DataType *B, DataType *C, int numARows,
             int numAColumns, int numBRows, int numBColumns) {
  for (int idy = 0; idy < numARows; ++idy) {
    for (int idx = 0; idx < numBColumns; ++idx) {
      for (int k = 0; k < numBRows; ++k) {
        C[idy * numBColumns + idx] +=
            A[idy * numAColumns + k] * B[k * numBColumns + idx];
      }
    }
  }
}

static void usage(char *prog) {
  printf("Usage: %s <numARows> <numAColumns> <numBColumns>\n"
         "Where all <..> are integer larger than zero.\n",
         prog);
}

#define ARGV_EXPECTED_LENGTH 4
int main(int argc, char **argv) {
  double time_start, time_elapsed;
  cudaError_t deviceError;
  DataType *hostA;     // The A matrix
  DataType *hostB;     // The B matrix
  DataType *hostC;     // The output C matrix
  DataType *resultRef; // The reference result
  DataType *deviceA;
  DataType *deviceB;
  DataType *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;
  int numCColumns;

  if (argc != ARGV_EXPECTED_LENGTH) {
    usage(argv[0]);
    return 1;
  }

  numARows = atoi(argv[1]);
  numAColumns = atoi(argv[2]);
  numBColumns = atoi(argv[3]);

  if (numARows < 1 || numAColumns < 1 || numBColumns < 1) {
    usage(argv[0]);
    return 1;
  }

  numBRows = numAColumns;
  numCRows = numARows;
  numCColumns = numBColumns;

  printf("numARows x numAColumns x numBColumns, cpu_exec (s), host_to_device (s), gpu_exec (s), device_to_host (s), differece\n");
  printf("%d x %d x %d, ", numARows , numAColumns , numBColumns);

  hostA = (DataType *)malloc(sizeof(DataType) * numARows * numAColumns);
  hostB = (DataType *)malloc(sizeof(DataType) * numBRows * numBColumns);
  hostC = (DataType *)malloc(sizeof(DataType) * numCRows * numCColumns);

  generateRandMatrix(hostA, numARows, numAColumns, 0, 1);
  generateRandMatrix(hostB, numBRows, numBColumns, 0, 1);
  memset(hostC, 0, sizeof(DataType) * numARows * numBColumns);

  time_start = getCpuSeconds();
  gemmCPU(hostA, hostB, hostC, numARows, numAColumns, numBRows, numBColumns);
  time_elapsed = getCpuSeconds() - time_start;
  printf("gemmCPU time: %lf s\n", time_elapsed);

  /* Allocate the needed Device memory*/
  deviceError = cudaMalloc(&deviceA, sizeof(DataType) * numARows * numAColumns);
  if (deviceError != cudaSuccess) {
    printf("Error when running GPU: %s (%d)\n", cudaGetErrorString(deviceError),
           deviceError);
  }
  deviceError = cudaMalloc(&deviceB, sizeof(DataType) * numBRows * numBColumns);
  if (deviceError != cudaSuccess) {
    printf("Error when running GPU: %s (%d)\n", cudaGetErrorString(deviceError),
           deviceError);
  }
  deviceError = cudaMalloc(&deviceC, sizeof(DataType) * numCRows * numCColumns);
  if (deviceError != cudaSuccess) {
    printf("Error when running GPU: %s (%d)\n", cudaGetErrorString(deviceError),
           deviceError);
  }

  /* Copy data to device.*/
  deviceError =
      cudaMemcpy(deviceA, hostA, sizeof(DataType) * numARows * numAColumns,
                 cudaMemcpyHostToDevice);
  if (deviceError != cudaSuccess) {
    printf("Error when running GPU: %s (%d)\n", cudaGetErrorString(deviceError),
           deviceError);
  }
  deviceError =
      cudaMemcpy(deviceB, hostB, sizeof(DataType) * numBRows * numBColumns,
                 cudaMemcpyHostToDevice);
  if (deviceError != cudaSuccess) {
    printf("Error when running GPU: %s (%d)\n", cudaGetErrorString(deviceError),
           deviceError);
  }
  deviceError =
      cudaMemset(deviceC, 0, sizeof(DataType) * numCRows * numCColumns);
  if (deviceError != cudaSuccess) {
    printf("Error when running GPU: %s (%d)\n", cudaGetErrorString(deviceError),
           deviceError);
  }

  dim3 block(BLOCK_X_SIZE, BLOCK_Y_SIZE);
  dim3 grid((numCColumns + BLOCK_X_SIZE - 1) / BLOCK_X_SIZE,
            (numCRows + BLOCK_Y_SIZE - 1) / BLOCK_Y_SIZE);

  time_start = getCpuSeconds();
  gemmGPU2D<<<grid, block>>>(deviceA, deviceB, deviceC, numARows, numAColumns,
                             numBRows, numBColumns);
  cudaDeviceSynchronize();
  time_elapsed = getCpuSeconds() - time_start;

  deviceError = cudaGetLastError();
  if (deviceError != cudaSuccess) {
    printf("Error when running GPU: %s (%d)\n", cudaGetErrorString(deviceError),
           deviceError);
  } else {
    printf("gemmGPU time: %lf s\n", time_elapsed);

    resultRef = (DataType *)malloc(sizeof(DataType) * numCRows * numCColumns);
    cudaMemcpy(resultRef, deviceC, sizeof(DataType) * numCRows * numCColumns,
               cudaMemcpyDeviceToHost);

    DataType diffEu =
        euclicianNormTwoMatrices(resultRef, hostC, numCRows, numCColumns);
    printf("Euclidian norm of the difference: %lf", diffEu);
  }

  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  free(hostA);
  free(hostB);
  free(hostC);
  free(resultRef);
  return 0;
}

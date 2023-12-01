#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define DataType double

#define TPW 32

/* Support function to generate a random number in a given range.
 * Based on the solution presented in the following forum thread:
 * http://ubuntuforums.org/showthread.php?t=1717717&p=10618266#post10618266*/
static DataType randfrom(DataType min, DataType max) {
  DataType range = (max - min);
  DataType div = RAND_MAX / range;
  return min + (rand() / div);
}

/* Populate a given vector memory location with values from a given range.*/
static void generateRandVector(DataType *matrix, int lenght, DataType min,
                        DataType max) {
  srand(time(NULL));

  for (int i = 0; i < lenght; ++i) {
    matrix[i] = randfrom(min, max);
  }
}

/*Get the current CPU time in seconds as a double.*/
static double getCpuSeconds(void) {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

/* Function to calculate the Euclidian norm on the difference between two vectors.*/
static DataType euclicianNormTwoVectors(DataType *vectorA, DataType *vectorB,
                                 int lenght) {
  DataType diffEu = 0;
  DataType *result = (DataType *)malloc(sizeof(DataType) * lenght);

  for (int i = 0; i < lenght; ++i) {
    result[i] = vectorA[i] - vectorB[i];
    diffEu += result[i] * result[i];
  }
  diffEu = sqrt(diffEu);

  free(result);
  return diffEu;
}

/* CUDA Kernel to add two vectors together.*/
__global__ void vecAddGPU(DataType *out, DataType *in1, DataType *in2,
                          int len) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx >= len)
    return;

  out[idx] = in1[idx] + in2[idx];
}

/*CPU function to add two vectors together.*/
static void vecAddCPU(DataType *out, DataType *in1, DataType *in2, int len) {
  for (int i = 0; i < len; ++i) {
    out[i] = in1[i] + in2[i];
  }
}

static void usage(char *prog) {
  printf("Usage: %s <inputLength>\n"
         "Where <inputLength> is an integer larger than zero.\n",
         prog);
}

int main(int argc, char **argv) {

  double time_start, time_elapsed;

  int inputLength;
  DataType *hostInput1;
  DataType *hostInput2;
  DataType *hostOutput;
  DataType *resultRef;
  DataType *deviceInput1;
  DataType *deviceInput2;
  DataType *deviceOutput;

  if (argc != 2) {
    usage(argv[0]);
    return 1;
  }

  inputLength = atoi(argv[1]);
  if (inputLength < 1) {
    usage(argv[0]);
    return 1;
  }

  printf("The input length is %d\n", inputLength);

  hostInput1 = (DataType *)malloc(sizeof(DataType) * inputLength);
  hostInput2 = (DataType *)malloc(sizeof(DataType) * inputLength);
  hostOutput = (DataType *)malloc(sizeof(DataType) * inputLength);

  generateRandVector(hostInput1, inputLength, 0, 1);
  generateRandVector(hostInput2, inputLength, 0, 1);

  time_start = getCpuSeconds();
  vecAddCPU(hostOutput, hostInput1, hostInput2, inputLength);
  time_elapsed = getCpuSeconds() - time_start;
  printf("vecAddCPU time: %lf s\n", time_elapsed);

  cudaMalloc(&deviceInput1, sizeof(DataType) * inputLength);
  cudaMalloc(&deviceInput2, sizeof(DataType) * inputLength);
  cudaMalloc(&deviceOutput, sizeof(DataType) * inputLength);

  cudaMemcpy(deviceInput1, hostInput1, sizeof(DataType) * inputLength,
             cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2, hostInput2, sizeof(DataType) * inputLength,
             cudaMemcpyHostToDevice);
  cudaMemset(deviceOutput, 0, sizeof(DataType) * inputLength);

  dim3 block(TPW, 1, 1);
  dim3 grid((inputLength + TPW - 1) / TPW, 1, 1);
  time_start = getCpuSeconds();
  vecAddGPU<<<grid, block>>>(deviceOutput, deviceInput1, deviceInput2,
                             inputLength);
  cudaDeviceSynchronize();
  time_elapsed = getCpuSeconds() - time_start;

  cudaError_t deviceError = cudaGetLastError();
  if (deviceError != cudaSuccess) {
    printf("Error when running GPU: %s (%d)\n", cudaGetErrorString(deviceError),
           deviceError);
  } else {
    printf("vecAddGPU time: %lf s\n", time_elapsed);

    resultRef = (DataType *)malloc(sizeof(DataType) * inputLength);
    cudaMemcpy(resultRef, deviceOutput, sizeof(DataType) * inputLength,
               cudaMemcpyDeviceToHost);

    DataType diffEu =
        euclicianNormTwoVectors(resultRef, hostOutput, inputLength);
    printf("Euclidian norm of the difference: %lf", diffEu);
  }

  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);
  free(resultRef);

  return 0;
}

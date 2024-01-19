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
static void generateRandVector(DataType *matrix, int length, DataType min,
                               DataType max) {
  srand(time(NULL));

  for (int i = 0; i < length; ++i) {
    matrix[i] = randfrom(min, max);
  }
}

/*Get the current CPU time in seconds as a double.*/
static double getCpuSeconds(void) {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

/* Function to calculate the Euclidian norm on the difference between two
 * vectors.*/
static DataType euclicianNormTwoVectors(DataType *vectorA, DataType *vectorB,
                                        int length) {
  DataType diffEu = 0;
  DataType *result = (DataType *)malloc(sizeof(DataType) * length);

  for (int i = 0; i < length; ++i) {
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
  printf("Usage: %s <inputLength> <segmentSize>\n"
         "Where <inputLength> is an integer larger than zero\n"
         "and segmentSize is an integer divisable by %d.\n",
         prog, TPW);
}

int main(int argc, char **argv) {

  double time_start, time_elapsed;
  cudaError_t deviceError;

  int inputLength;
  int segmentSize;
  int numberOfSegments;
  DataType *hostInput1;
  DataType *hostInput2;
  DataType *hostOutput;
  DataType *resultRef;
  DataType *deviceInput1;
  DataType *deviceInput2;
  DataType *deviceOutput;

  if (argc != 3) {
    usage(argv[0]);
    return 1;
  }

  inputLength = atoi(argv[1]);
  segmentSize = atoi(argv[2]);
  if (inputLength < 1) {
    usage(argv[0]);
    return 1;
  }

  if ((segmentSize % TPW) != 0) {
    usage(argv[0]);
    return 1;
  }

  numberOfSegments = (inputLength + (segmentSize - 1)) / segmentSize;

  printf("inputLength, segmentSize, cpu_exec (s), gpu_exec (s), differece\n");
  printf("%d, %d, ", inputLength, segmentSize);

  deviceError =
      cudaHostAlloc((void **)&hostInput1, sizeof(DataType) * inputLength,
                    cudaHostAllocDefault);
  if (deviceError != cudaSuccess) {
    printf("CUDA Error: %s (%d)\n", cudaGetErrorString(deviceError),
           deviceError);
  }

  deviceError =
      cudaHostAlloc((void **)&hostInput2, sizeof(DataType) * inputLength,
                    cudaHostAllocDefault);
  if (deviceError != cudaSuccess) {
    printf("CUDA Error: %s (%d)\n", cudaGetErrorString(deviceError),
           deviceError);
  }
  deviceError =
      cudaHostAlloc((void **)&resultRef, sizeof(DataType) * inputLength,
                    cudaHostAllocDefault);
  if (deviceError != cudaSuccess) {
    printf("CUDA Error: %s (%d)\n", cudaGetErrorString(deviceError),
           deviceError);
  }
  hostOutput = (DataType *)malloc(sizeof(DataType) * inputLength);

  generateRandVector(hostInput1, inputLength, 0, 1);
  generateRandVector(hostInput2, inputLength, 0, 1);

  time_start = getCpuSeconds();
  vecAddCPU(hostOutput, hostInput1, hostInput2, inputLength);
  time_elapsed = getCpuSeconds() - time_start;
  printf("%lf, ", time_elapsed);

  deviceError = cudaMalloc(&deviceInput1, sizeof(DataType) * inputLength);
  if (deviceError != cudaSuccess) {
    printf("CUDA Error: %s (%d)\n", cudaGetErrorString(deviceError),
           deviceError);
  }
  deviceError = cudaMalloc(&deviceInput2, sizeof(DataType) * inputLength);
  if (deviceError != cudaSuccess) {
    printf("CUDA Error: %s (%d)\n", cudaGetErrorString(deviceError),
           deviceError);
  }
  deviceError = cudaMalloc(&deviceOutput, sizeof(DataType) * inputLength);
  if (deviceError != cudaSuccess) {
    printf("CUDA Error: %s (%d)\n", cudaGetErrorString(deviceError),
           deviceError);
  }

  dim3 block(TPW, 1, 1);
  dim3 grid((segmentSize + TPW - 1) / TPW, 1, 1);

  #define NUMBER_OF_STREAMS 4
  cudaStream_t streams[NUMBER_OF_STREAMS];
  int inputLength_stream = segmentSize;
  int stream_id = 0;

  time_start = getCpuSeconds();
  for (int i = 0; i < inputLength; i += segmentSize) {
    cudaStreamCreate(&streams[stream_id]);

    if ((i + segmentSize) > inputLength) {
      inputLength_stream = inputLength - i;
    }

    cudaMemcpyAsync(&deviceInput1[i], &hostInput1[i],
                    sizeof(DataType) * inputLength_stream, cudaMemcpyHostToDevice,
                    streams[stream_id]);

    cudaMemcpyAsync(&deviceInput2[i], &hostInput2[i],
                    sizeof(DataType) * inputLength_stream, cudaMemcpyHostToDevice,
                    streams[stream_id]);

    cudaMemsetAsync(&deviceOutput[i], 0, sizeof(DataType) * inputLength_stream,
                    streams[stream_id]);

    vecAddGPU<<<grid, block, 0, streams[stream_id]>>>(
        &deviceOutput[i], &deviceInput1[i], &deviceInput2[i],
        inputLength_stream);

    cudaMemcpyAsync(&resultRef[i], &deviceOutput[i],
                    sizeof(DataType) * inputLength_stream,
                    cudaMemcpyDeviceToHost, streams[stream_id]);

    stream_id += 1;
    if (stream_id >= NUMBER_OF_STREAMS)
      stream_id = 0;
  }

  for (int i = 0; i < NUMBER_OF_STREAMS; ++i) {
    cudaStreamSynchronize(streams[i]);
    cudaStreamDestroy(streams[i]);
  }
  time_elapsed = getCpuSeconds() - time_start;
  printf("%lf, ", time_elapsed);

  deviceError = cudaGetLastError();
  if (deviceError != cudaSuccess) {
    printf("Error when running GPU: %s (%d)\n", cudaGetErrorString(deviceError),
           deviceError);
  }

  DataType diffEu = euclicianNormTwoVectors(resultRef, hostOutput, inputLength);
  printf("%lf\n", diffEu);

  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);

  cudaFreeHost(hostInput1);
  cudaFreeHost(hostInput2);
  cudaFreeHost(resultRef);

  free(hostOutput);

  return 0;
}

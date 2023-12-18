
#include <random>
#include <stdio.h>
#include <sys/time.h>

#define NUM_BINS 4096
#define TPB 1024

#define SATURATION 127

/* Cuda kernel to calculate histogram bins based on an input vector.*/
__global__ void histogram_kernel(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {
  const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

  __shared__ unsigned int local_bins[NUM_BINS];

  /* Source: https://stackoverflow.com/a/6487821 */
  for (unsigned int i = threadIdx.x; i < num_bins; i += blockDim.x) {
    local_bins[i] = 0;
  }
  __syncthreads();

  if (idx < num_elements) {
    unsigned int index = input[idx];
    if (index < num_bins) {
      atomicAdd(&local_bins[index], 1);
    }
  }
  __syncthreads();

  for (unsigned int i = threadIdx.x; i < num_bins; i += blockDim.x) {
    atomicAdd(&bins[i], local_bins[i]);
  }
}

/* Cuda kernel to set all values of a vector to a max value.*/
__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins) {

  const int idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx < num_bins) {
    if (bins[idx] > SATURATION) {
      bins[idx] = SATURATION;
    }
  }
}

/* Function to calculate the Euclidian norm on the difference between two
 * vectors.*/
static double euclicianNormTwoVectors(unsigned int *vectorA,
                                      unsigned int *vectorB, int length) {
  double diffEu = 0;
  double *result = (double *)malloc(sizeof(double) * length);

  for (int i = 0; i < length; ++i) {
    result[i] = (double)(vectorA[i] - vectorB[i]);
    diffEu += result[i] * result[i];
  }
  diffEu = sqrt(diffEu);

  free(result);
  return diffEu;
}

/* Calculate a histogram on the CPU.*/
void histogram_cpu(unsigned int *input, unsigned int *bins,
                   unsigned int num_elements, unsigned int num_bins) {

  for (int i = 0; i < num_elements; ++i) {
    unsigned int index = input[i];
    if (index < num_bins) {
      if (bins[index] < 127) {
        bins[index] += 1;
      }
    } else {
      printf("ERROR: Incorrect value (%u) at location %d\n", input[i], i);
    }
  }
}

/*Get the current CPU time in seconds as a double.*/
static double getCpuSeconds(void) {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

/* Support function to generate a random number in a given range.
 * Based on the solution presented in the following forum thread:
 * http://ubuntuforums.org/showthread.php?t=1717717&p=10618266#post10618266*/
static unsigned int randfrom(unsigned int min, unsigned int max) {
  unsigned int range = (max - min);
  unsigned int div = RAND_MAX / range;
  return min + (rand() / div);
}

/* Populate a given vector memory location with values from a given range.*/
static void generateRandVector(unsigned int *vector, int length,
                               unsigned int min, unsigned int max) {
  srand(time(NULL));

  for (int i = 0; i < length; ++i) {
    vector[i] = randfrom(min, max);
  }
}

static void usage(char *prog) {
  printf("Usage: %s <inputLength>\n"
         "Where <inputLength> is an integer larger than zero.\n",
         prog);
}

int main(int argc, char **argv) {
  double time_start, time_elapsed;
  cudaError_t deviceError;

  unsigned int inputLength;
  unsigned int *hostInput;
  unsigned int *hostBins;
  unsigned int *resultRef;
  unsigned int *deviceInput;
  unsigned int *deviceBins;

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

  hostInput = (unsigned int *)malloc(sizeof(unsigned int) * inputLength);
  hostBins = (unsigned int *)malloc(sizeof(unsigned int) * NUM_BINS);

  generateRandVector(hostInput, inputLength, 0, (NUM_BINS - 1));

  histogram_cpu(hostInput, hostBins, inputLength, NUM_BINS);

  deviceError = cudaMalloc(&deviceBins, sizeof(unsigned int) * NUM_BINS);
  if (deviceError != cudaSuccess) {
    printf("Error when allocating memory in GPU: %s (%d)\n",
           cudaGetErrorString(deviceError), deviceError);
  }
  deviceError = cudaMalloc(&deviceInput, sizeof(unsigned int) * inputLength);
  if (deviceError != cudaSuccess) {
    printf("Error when allocating memory in GPU: %s (%d)\n",
           cudaGetErrorString(deviceError), deviceError);
  }
  deviceError =
      cudaMemcpy(deviceInput, hostInput, sizeof(unsigned int) * inputLength,
                 cudaMemcpyHostToDevice);
  if (deviceError != cudaSuccess) {
    printf("Error when copying data to GPU: %s (%d)\n",
           cudaGetErrorString(deviceError), deviceError);
  }
  deviceError = cudaMemset(deviceBins, 0, sizeof(unsigned int) * NUM_BINS);
  if (deviceError != cudaSuccess) {
    printf("Error when setting memory to value in GPU: %s (%d)\n",
           cudaGetErrorString(deviceError), deviceError);
  }

  dim3 hist_block(TPB, 1, 1);
  dim3 hist_grid((inputLength + TPB - 1) / TPB, 1, 1);

  time_start = getCpuSeconds();
  histogram_kernel<<<hist_grid, hist_block>>>(deviceInput, deviceBins,
                                              inputLength, NUM_BINS);
  cudaDeviceSynchronize();
  deviceError = cudaGetLastError();
  if (deviceError != cudaSuccess) {
    printf("Error when running GPU: %s (%d)\n", cudaGetErrorString(deviceError),
           deviceError);
  }

  dim3 conv_block(TPB, 1, 1);
  dim3 conv_grid((NUM_BINS + TPB - 1) / TPB, 1, 1);

  convert_kernel<<<conv_grid, conv_block>>>(deviceBins, NUM_BINS);
  cudaDeviceSynchronize();
  time_elapsed = getCpuSeconds() - time_start;
  printf("Total GPU time: %lf (s)\n", time_elapsed);

  deviceError = cudaGetLastError();
  if (deviceError != cudaSuccess) {
    printf("Error when running GPU: %s (%d)\n", cudaGetErrorString(deviceError),
           deviceError);
  } else {
    resultRef = (unsigned int *)malloc(sizeof(unsigned int) * NUM_BINS);

    cudaMemcpy(resultRef, deviceBins, sizeof(unsigned int) * NUM_BINS,
               cudaMemcpyDeviceToHost);

    double diffEu = euclicianNormTwoVectors(resultRef, hostBins, NUM_BINS);
    printf("Euclidian norm: %lf\n", diffEu);
  }

  cudaFree(deviceInput);
  cudaFree(deviceBins);

  free(hostInput);
  free(hostBins);
  free(resultRef);

  return 0;
}

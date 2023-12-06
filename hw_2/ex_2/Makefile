NVCC ?= nvcc

NVCCFLAGS ?= -arch=sm_70

FLAGS ?=

BUILD_DIR ?= ./

all: $(BUILD_DIR)/matrixMult

$(BUILD_DIR)/%: $(BUILD_DIR)/%.o
	$(NVCC) $(NVCCFLAGS) $(FLAGS) $^ -o $@

$(BUILD_DIR)/%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(FLAGS) -c $< -o $@

NVCC ?= nvcc

NVCCFLAGS ?= -arch=sm_70

BUILD_DIR ?= ./

all: $(BUILD_DIR)/vectorAdd
	cp run $(BUILD_DIR)/run-vectorAdd

$(BUILD_DIR)/%: $(BUILD_DIR)/%.o
	$(NVCC) $(NVCCFLAGS) $^ -o $@

$(BUILD_DIR)/%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

NVCC ?= nvcc

NVCCFLAGS ?= -arch=sm_70
NVCCFLAGS += -lcublas
NVCCFLAGS += -lcusparse

BUILD_DIR ?= ./

all: $(BUILD_DIR)/heatEq

$(BUILD_DIR)/%: $(BUILD_DIR)/%.o
	$(NVCC) $(NVCCFLAGS) $^ -o $@

$(BUILD_DIR)/%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

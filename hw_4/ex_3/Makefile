NVCC ?= nvcc

NVCCFLAGS ?= -arch=sm_70
NVCCFLAGS += -lcublas
NVCCFLAGS += -lcusparse

BUILD_DIR ?= ./

all: $(BUILD_DIR)/heatEq
	cp run $(BUILD_DIR)/run-heatEq

noconcurr: NVCCFLAGS += -DNO_CONCURRENT_MEM=1
noconcurr: $(BUILD_DIR)/heatEq
	cp run $(BUILD_DIR)/run-heatEq

$(BUILD_DIR)/%: $(BUILD_DIR)/%.o
	$(NVCC) $(NVCCFLAGS) $^ -o $@

$(BUILD_DIR)/%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

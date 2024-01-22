export BUILD_DIR = $(abspath ./)

all: hw_2

hw_2: vectorAdd matrixMult histogram

vectorAdd:
	$(MAKE) -C hw_2/ex_1

matrixMult:
	$(MAKE) -C hw_2/ex_2

matrixMult-float:
	$(MAKE) -C hw_2/ex_2 FLAGS=-DUSE_FLOAT

vectorAdd-stream:
	$(MAKE) -C hw_4/ex_2

histogram:
	$(MAKE) -C hw_3/ex_1

heatEq:
	$(MAKE) -C hw_4/ex_3


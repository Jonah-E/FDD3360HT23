export BUILD_DIR = $(abspath ./)

all: hw_2

hw_2: vectorAdd matrixMult

vectorAdd:
	$(MAKE) -C hw_2/ex_1

matrixMult:
	$(MAKE) -C hw_2/ex_2

matrixMult-float:
	$(MAKE) -C hw_2/ex_2 FLAGS=-DUSE_FLOAT


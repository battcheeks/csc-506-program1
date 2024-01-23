# Location of the CUDA Toolkit
NVCC := $(CUDA_PATH)/bin/nvcc

build: plot1

plot1.o:plot1.cu
	$(NVCC) $(INCLUDES) $(GENCODE_FLAGS) -o $@ -c $<

plot1: plot1.o
	$(NVCC) $(LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

run: build
	$(EXEC) ./plot1

clean:
	rm -f plot1 *.o

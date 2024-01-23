# Location of the CUDA Toolkit
NVCC := $(CUDA_PATH)/bin/nvcc

build: plot2

plot2.o:plot2.cu
	$(NVCC) $(INCLUDES) $(GENCODE_FLAGS) -o $@ -c $<

plot2: plot2.o
	$(NVCC) $(LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

run: build
	$(EXEC) ./plot2

clean:
	rm -f plot2 *.o

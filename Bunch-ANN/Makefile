SDK_INSTALL_PATH := /usr/local/cuda
NVCC=$(SDK_INSTALL_PATH)/bin/nvcc
LIB       :=  -L$(SDK_INSTALL_PATH)/lib64 -L$(SDK_INSTALL_PATH)/samples/common/lib/linux/x86_64
#INCLUDES  :=  -I$(SDK_INSTALL_PATH)/include -I$(SDK_INSTALL_PATH)/samples/common/inc
OPTIONS   :=  -O3 #-std=c99 -lm
#--maxrregcount=100 --ptxas-options -v 

EXECS :=  bpl_CPU bpl_GPU
all: $(EXECS) run

clean:
	rm -f $(EXECS) *.o

run:
	clear
	./bpl_GPU -N 16 -V -P 4 -M 4 -S 8 -I 2
	#./bpl_CPU -N 16 -P 4 -M 4 -S 8 -I 2
	#./test

timer.o: timer.cu timer.h
	${NVCC} $< -c -o $@ $(OPTIONS)

#Example make commands.
#Modify to include the Kernels that you create
bplKernel.o: bplKernel.cu bplKernel.h 
	${NVCC} $< -c -o $@ $(OPTIONS)

bpl_GPU: bpl.cu timer.o #bplKernel.h bplKernel.o
	#${NVCC} $< bplKernel.o -o $@ $(LIB) timer.o $(OPTIONS)
	${NVCC} $< -o $@ $(LIB) timer.o $(OPTIONS)

bpl_CPU: bpl_cpu.cu timer.o
	${NVCC} $< -o $@ $(LIB) timer.o $(OPTIONS)

#test: test.cu
#	${NVCC} $< -o $@ $(LIB) $(OPTIONS)

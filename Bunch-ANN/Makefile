SDK_INSTALL_PATH := /usr/local/cuda
NVCC=$(SDK_INSTALL_PATH)/bin/nvcc
LIB       :=  -L$(SDK_INSTALL_PATH)/lib64 -L$(SDK_INSTALL_PATH)/samples/common/lib/linux/x86_64
#INCLUDES  :=  -I$(SDK_INSTALL_PATH)/include -I$(SDK_INSTALL_PATH)/samples/common/inc
OPTIONS   :=  -O3 --ptxas-options -v --gpu-architecture=sm_61 --compiler-bindir /usr/local/gcc-6.4.0/bin -std=c++11 -I/s/bach/c/under/joshtb/cuda-patches/include #-std=c99 -lm 
#--maxrregcount=100 --ptxas-options -v 

EXECS :=  bpl_CPU bpl_GPU
all: $(EXECS) run

clean:
	rm -f $(EXECS) *.o

run:
	clear
	./bpl_GPU -N 4 -V -P 4 -M 4 -S 8 -I 2
	#./bpl_CPU -N 16 -P 4 -M 4 -S 8 -I 2
	#./test

timer.o: timer.cu timer.h
	${NVCC} $< -c -o $@ $(OPTIONS)

#Example make commands.
#Modify to include the Kernels that you create
bunch_ann.o: bunch_ann.cu #bplKernel.h 
	${NVCC} $< -c -o $@ $(OPTIONS)

bpl_GPU: bpl.cu timer.o #bunch_ann.o #bplKernel.h bplKernel.o
	#${NVCC} bunch_ann.cu -c -o bunch_ann.o $(LIB) $(OPTIONS)
	${NVCC} bunch_ann.cu -c -o bunch_ann.o ${LIB} $(OPTIONS)
	${NVCC} $< -o $@ $(LIB) bunch_ann.o timer.o $(OPTIONS)

bpl_CPU: bpl_cpu.cu timer.o
	${NVCC} $< -o $@ $(LIB) timer.o $(OPTIONS)

#test: test.cu
#	${NVCC} $< -o $@ $(LIB) $(OPTIONS)

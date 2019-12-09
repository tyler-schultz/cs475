SDK_INSTALL_PATH :=  /usr/local/cuda
NVCC=$(SDK_INSTALL_PATH)/bin/nvcc
LIB       :=  -L$(SDK_INSTALL_PATH)/lib64 -L$(SDK_INSTALL_PATH)/samples/common/lib/linux/x86_64
#INCLUDES  :=  -I$(SDK_INSTALL_PATH)/include -I$(SDK_INSTALL_PATH)/samples/common/inc
OPTIONS   :=  -O3 #-std=c99 -lm
#--maxrregcount=100 --ptxas-options -v 

EXECS :=  bpl_CPU #bpl_GPU
all:$(EXECS)

clean:
	rm -f $(EXECS) *.o

timer.o : timer.cu timer.h
	${NVCC} $< -c -o $@ $(OPTIONS)

#Example make commands.
#Modify to include the Kernels that you create
#bplKernel.o : bplKernel.cu bplKernel.h 
#	${NVCC} $< -c -o $@ $(OPTIONS)

#bpl_GPU : bpl.cu  bplKernel.h bplKernel.o timer.o
#	${NVCC} $< bplKernel.o -o $@ $(LIB) timer.o $(OPTIONS)

bpl_CPU : bpl_cpu.cu timer.o
	${NVCC} $< -o $@ $(LIB) timer.o $(OPTIONS)
  




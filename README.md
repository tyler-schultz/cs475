**README:** Files for the final project, CS475 Fall 2019

**Author:** Joshua Burris, Tyler Schultz, Tristen Gulley-Davenport

**Date:** 12/12/2019


Objective is to 
(i)   Implement bunch-mode backpropagation learning(BPL) on GPUS using CUDA. 
(ii)  Study the performance(execution time) with varying bunch size.
Here, bunch size means the number of samples trained together in a single pass. 
(iii) (Optional) Study the trade-off of model accuracy versus the gains of training 
time that we obtain from bunch BPL. 

The reference code has been created by Paul Tero at Existor Ltd 
as part of a neural networks tutorial. The original website seems
to be broken. Here is a link to the snapshot of the page. 

<http://webcache.googleusercontent.com/search?q=cache:Hw1TbvXkLtwJ:
www.existor.com/en/ml-neural-network-code.html+&cd=1&hl=en&ct=clnk&gl=in> 

Modifications were made to work for dynamically allocated data-structures
and also for bunch-mode BPL.

This may not be the best sequential code for CPU implementation because:
1. The matrix multiplication routines are implemented by hand. Using Intel's
	MKL library routines achieve the best performance.
2. Certain data-structures can be modified in order to make use of locality
	Example: The input(X) matrix can be transposed. This can avoid the 
		memcpy in the code. 

So, the speed-up that you report for GPUs may not be universally true for
all the back-propagation learning algorithms. It will be with respect to this
particular CPU code.

1.Contents
  	
	Reference/
		ann-bunch.pdf  : A research paper that implements ANN training
				for Speech Recognition using bunch BPL algorithm
				on GPUs
		neuralnetwork.c: A reference code that implements a simple ANN 
				training of 3 samples that has 3 inputs and 3 
				outputs	using BPL algorithm. It uses a single 
				hidden layer with 3 neurons . It uses hyperbolic
				tan as the activation function and the softmax
				function at the output layer.  		
	
	Bunch-ANN/
		bpl_cpu.cu	:Includes the main function where the sequential 
				training is implemented to run on the host (CPU)
		bpl_gpu.cu		:Same as bpl_cpu.cu. Modify to implement parallelized 
						 function.
		bunch_ann_cpu.h	:Includes routines that are used in the main program.
					Example: matrix multiplication, activation
				 	function and its gradient, softmax function etc.
		bunch_ann_gpu.h	:(Similar to bunch_ann_cpu.h) Includes routines used by the main
					program that have integrated to use CUDA kernels.
		Makefile	:Makes the files to run on GPUs using NVCC compiler
		timer.cu	:Used to measure the running time of the program
		timer.h		:Supporting file for timer.c
		util_cpu.h	:Includes utility functions such as command line
				 	parsing and printing of variables.
		util_gpu.h	:(Similar to util_cpu.h) Includes utility functions such as command line
				 	parsing and printing of variables, but implemented for single
					pointer doubles as 2D arrays.

	Bunch-ANN-flatten/
		bpl_flatten.cu  :Same as bpl_cpu.cu except that the 2d arrays are flattened into 1 dimension.
		bunch-new.c	:Supporting host function calls. (Similar to bunch_ann.h to accomodate the changes 
				 for flattened array.)
		Makefile	:Makes the files to run on GPUs using NVCC compiler.
		timer.cu	:Used to measure the running time of the program.
		timer.h		:Supporting file for timer.c
		util.h		:Includes utility functions such as command line
				 		parsing and printing of variables. 
		
		
		
### 2. Usage
	
	Once you make the files in the respective folders, running the executable with -H option
	gives the following:
	./bpl_CPU -H
	usage: ./bpl_CPU
		-N Number of input layer neurons
		-M Number of hidden layer neurons
		-P Number of output layer neurons
		-S Number of training samples
		-I Number of training samples per bunch
		-L Number of iterations of time loop
		-V Verbose ON
		-H usage help, this dialogue

	The N,M,P,S,I,L options require arguments that are greater than or equal to 1.
	Note: Number of training samples per iteration (I) should be less than or equal to 
		Number of training samples (S)

3. Example input/output

	A ANN with 4 neurons(M) of 1 hidden layer, 30(S) samples each with 1 input(N) and
	5(P) outputs. The bunch-size is 5(I) and the number of time loop iterations is 10(L)
 
	With verbose:
	$./bpl_CPU -N1 -M4 -S30 -I5 -L10 -P5 -v
	Time: 0.000351

	input/hidden weights:
	   0.07025	  -0.02358	   0.05504	   0.06093	
	   0.08233	  -0.06049	  -0.03296	   0.05365	

	hidden/output weights:
	  -0.03527	   0.01972	   0.00444	   0.03458	  -0.01791	
	   0.00331	   0.09106	   0.08386	   0.02775	   0.04409	
	  -0.07188	   0.02119	  -0.09694	  -0.05162	  -0.07276	
	   0.06135	  -0.06817	  -0.01931	  -0.07355	  -0.07773	
	   0.10034	  -0.05581	   0.00313	   0.06835	   0.02308	
	
	Without verbose:
	$ ./bpl_CPU -N1 -M4 -S30 -I5 -L10 -P5 
	Time: 0.000364

4. Code Description

	The bpl_cpu.cu has the main function. It is divided into 7 portions:
	(i)   Command line parsing : A structure is created to hold all the command 
				     line arguments. 
	(ii)  Variable Declaration : The required matrices are defined. 
				     Learning rate is a parameter to be mulitplied 
				     to the weight matrices when back propagating 
				     the errors.	 
				     k2,k3 variables are used to account for partial 
				     bunch sizes. If the bunch size is a mulitple
				     of total of samples, then k3=0.
				     Example: S=30, I=5, K2=6,   K3=0
					      S=30, I=4, K2=7,   K3=(30-(7*4))=2 
				     This means that when you have I=5, you are equally 
				     diving the training samples into 6(K2) bunches. 
				     When you have I=4, the first 7(K2) bunches have a size
				     of 4(I) whereas the last bunch has a size 2 (K3).
	(iii) Memory allocations   : Corresponding memory allocations for the pointers.
	(iv)  Initializations	   : Weights, inputs and outputs are initialized
	(v)   Training		   : It has 2 iterations. One is over the time loop. 
				     Other is over the bunches. Notice the bunch loop is 
				     divided into 2 to account for the partial bunches.
				     Inside the 2 loops, we have 3 phases.
					1. Initialization: X[0]=H[0]=1 is the bias or the 
						threshold. X and Y matrices are updated 
						from inputs and output matrices respectively.
					2. Forward Phase: It multiplies the input(X) and 
						input-hidden weight matrix (Wxh) to produce the
						(Zh) matrix which is then acted upon by an 
						activation function(tanh) to produce hidden
						(H) matrix. This is then mulitplied with 
						hidden-output weight(Why) matrix to produce 
						the output(Zy). This output is then acted upon
						by the softmax function to produce the 
						predicted output(P1). The forward phase	ends with 
						finding	the error matrix(E) as the difference 
						of P1 and the expected output(Y).
					3. Backpropagation phase: The Why and Whx matrices are
						updated by backpropogating the errors. 
						Example: Suppose A=B*C. We can find B'=A*C"
						Here C" -> means transpose of C. Here B and B'
						does not have same values but have same dimension.
						So, starting from our error matrix (E), we can get
						error hidden-output weight matrix (dWhy) by mulitplying
						it with (H).This is then subtracted with the
						previous hidden-output weight matrix matrix(Why).
						From the new Why, we can update H, Zh and Wxh 
						in a similar way. 
					For the further iterations, we retain only the values of 
					X,Y(original) and Why,Whx(updated). 
					Remaining should be computed from the beginning. Be careful
					not to add/subtract with the values from the previous iteration. 				    
	(vi)   Print outputs	   :Calls the print function in util.h
	(vii)  Free Memory	   :Dynamically allocated memory is freed.

/*---------------------------------------------------------------------------------------------------------------*/
/// bpl.c
/// For CSU CS475 Fall 2016
/// Instructor: Sanjay Rajopadhye
/// GTA: Swetha Varadarajan
/// Based on code Created by Paul Tero at Existor Ltd as part of a neural networks tutorial
/// Modified by Swetha Varadarajan
/// Created: 2016-11-16
/*---------------------------------------------------------------------------------------------------------------*/

#include <stdio.h> 
#include <stdlib.h> 
#include <string.h> 
#include <math.h> 

#include "timer.h"
#include "util.h"
#include "bunch_ann.h"

int main(int argc, char** argv) 
{

/*---------------------------------------------------------------------------------------------------------------*/
/*-----------------------------------------Command line parsing--------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------*/

  Params cmdLineArgs;
  parseCmdLineArgs(&cmdLineArgs,argc,argv);

/*---------------------------------------------------------------------------------------------------------------*/
/*-------------------------------------------Variable Declaration------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------*/

  /*Array description and its size in the comments next to its declation*/

  double **inputs;//Given inputs = total number of samples(S)*number of inputs per sample(N) 
  double **outputs;//Expected outputs = total number of samples(S)*number of outputs per sample(P) 

  double **X;//Input for a given iteration = bunch size(I)*number of inputs per sample(N+1(bias))
  double **Y;//Output for a given iteration = bunch size(I)*number of outputs per sample(P)

  double **Wxh; //Weights in between input and hidden layer = (N+1)*M
  double **Why; //Weights in between input and hidden layer = (M+1)*P
  double **dWxh; //Error Weights in between input and hidden layer = (N+1)*M
  double **dWhy; //Error Weights in between input and hidden layer = (M+1)*P

  double **Zh; //Weighted sum for hidden layer=I*M
  double **H;  // Activation values = I*(M+1)
  double **Zy; //Weighted sum for output layer=I*P 
  double **E;  //Calculated Errors = I*P
  double **P1; //Oredicted output = I*P
  double **P;  // (exp(Zy)) = I*P
  double *sum; //(summation of the P[i]s) = I
  
  double learningrate = 0.0001; /*learning rate */
  long b = cmdLineArgs.sample_per_iter;
  
  long k2 = cmdLineArgs.sample_total/b ; /*number of full bunches */
  long k3 = cmdLineArgs.sample_total-(k2*b); /* size of the partial bunch */

/*---------------------------------------------------------------------------------------------------------------*/
/*-------------------------------------------Memory allocations--------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------*/
 
  inputs  = (double**)malloc(cmdLineArgs.sample_total * sizeof(double*));
  outputs = (double**)malloc(cmdLineArgs.sample_total * sizeof(double*));
  
  sum	  = (double*)malloc((b)*sizeof(double));

  for(long i = 0; i < cmdLineArgs.sample_total; ++i )
  {
	inputs[i] =(double*)malloc(cmdLineArgs.N * sizeof(double));
	outputs[i]=(double*)malloc(cmdLineArgs.P * sizeof(double));
  }

  Wxh     = (double**)malloc((cmdLineArgs.N+1) * sizeof(double*));
  Why	  = (double**)malloc((cmdLineArgs.M+1) * sizeof(double*));
  dWxh    = (double**)malloc((cmdLineArgs.N+1) * sizeof(double*));
  dWhy	  = (double**)malloc((cmdLineArgs.M+1) * sizeof(double*));

  for(long i = 0; i < cmdLineArgs.N+1; ++i )
  {
	Wxh[i] =(double*)malloc(cmdLineArgs.M * sizeof(double));	
	dWxh[i]=(double*)malloc(cmdLineArgs.M * sizeof(double));
  }

  for(long i = 0; i < cmdLineArgs.M+1; ++i )
  {
	Why[i] =(double*)malloc(cmdLineArgs.P * sizeof(double));
	dWhy[i]=(double*)malloc(cmdLineArgs.P * sizeof(double));
  }

  X	  = (double**)malloc(b*sizeof(double*));
  E	  = (double**)malloc(b*sizeof(double*));
  P	  = (double**)malloc(b*sizeof(double*));
  P1  = (double**)malloc(b*sizeof(double*));
  H	  = (double**)malloc(b*sizeof(double*));
  Zh  = (double**)malloc(b*sizeof(double*));
  Zy  = (double**)malloc(b*sizeof(double*));

  for(long i = 0; i < b; ++i )
  {
  X[i]	  = (double*)malloc((cmdLineArgs.N+1)*sizeof(double));
  E[i]	  = (double*)malloc(cmdLineArgs.P*sizeof(double));
  P[i]	  = (double*)malloc(cmdLineArgs.P*sizeof(double));
  P1[i]    = (double*)malloc(cmdLineArgs.P*sizeof(double));
  H[i]	  = (double*)malloc((cmdLineArgs.M+1)*sizeof(double));
  Zh[i]	  = (double*)malloc(cmdLineArgs.M*sizeof(double));
  Zy[i]	  = (double*)malloc(cmdLineArgs.P*sizeof(double));
  }

  if( inputs == NULL || outputs == NULL || X == NULL|| H == NULL || dWxh == NULL || dWhy == NULL 
      || Zh == NULL || Zy == NULL || Wxh == NULL || Why == NULL|| E == NULL || P == NULL
	  || P1 == NULL || sum == NULL)
  {
    printf( "Could not allocate memory\n" );
    exit(0);
  }
/*---------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------Initializations--------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------*/

  initializeW(Wxh,(cmdLineArgs.N+1),cmdLineArgs.M);
  initializeW(Why,(cmdLineArgs.M+1),cmdLineArgs.P);
  initializeI(inputs,cmdLineArgs.sample_total,cmdLineArgs.N);
  initializeO(outputs,cmdLineArgs.sample_total,cmdLineArgs.P);

/*---------------------------------------------------------------------------------------------------------------*/
/*------------------------------------------------Training-------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------*/
  initialize_timer();
  start_timer();
  for (long t=0; t<cmdLineArgs.iter; t++) //Time loop
  {
	 for (long s=0; s<k2; s++) //Bunch loop
	  { 	
		for(long i=0;i<b;i++)
		{
		X[i][0]=H[i][0]=1;//bias setting
		//required input/output are copied from inputs/outputs to X and Y
	 	memcpy (&X[i][1], inputs[(s*b)+i], cmdLineArgs.N*sizeof(double)); 
		}
		Y = &outputs[s*b]; 

		/*Forward Phase*/
		mm(Zh,X,Wxh,b,cmdLineArgs.N+1,cmdLineArgs.M); //Zh=X*Wxh
		func(H,Zh,b,cmdLineArgs.M,1); //H=f1(Zh)
		mm(Zy,H,Why,b,cmdLineArgs.M+1,cmdLineArgs.P); //Zy=H*Why	
		func(P,Zy,b,cmdLineArgs.P,0); //P=fn(Zy)	
		reduction(P,sum,b,cmdLineArgs.P);  //summation of probabilities for each training sample
		prob(P,P1,sum,b,cmdLineArgs.P); //P1=fn(P,sum)	
		error(E,P1,Y,b,cmdLineArgs.P);	//E=P1-Y

		/*Backprpagation Phase*/ 		
		mtm(dWhy,H,E,cmdLineArgs.M+1,b,cmdLineArgs.P); //dWhy=H'*E ('->transpose)		
		delta(Why,dWhy,cmdLineArgs.M+1,cmdLineArgs.P,learningrate); //Why=fn(dwhy)
		mmt(H,Why,E,b,cmdLineArgs.M+1,cmdLineArgs.P); //H=Why*E'		
		gradient_func(Zh,H,b,cmdLineArgs.M); //Zh=f1"(H) ("->gradient of f1)		
		mtm(dWxh,X,Zh,cmdLineArgs.N+1,b,cmdLineArgs.M);	//dWxh=X'Zh
		delta(Wxh,dWxh,cmdLineArgs.N+1,cmdLineArgs.M,learningrate);//Wxh=fn(dWxh)
	}
	if(k3)
	{
		for(long i=0;i<k3;i++)
		{
		X[i][0]=H[i][0]=1;
	 	memcpy (&X[i][1], inputs[(k2*b)+i], cmdLineArgs.N*sizeof(double));
		}
		Y = &outputs[k2*b];

		/*Forward Phase*/
		mm(Zh,X,Wxh,k3,cmdLineArgs.N+1,cmdLineArgs.M);
		func(H,Zh,k3,cmdLineArgs.M,1);
		mm(Zy,H,Why,k3,cmdLineArgs.M+1,cmdLineArgs.P);		
		func(P,Zy,k3,cmdLineArgs.P,0); 
		reduction(P,sum,k3,cmdLineArgs.P);  
		prob(P,P1,sum,k3,cmdLineArgs.P);  
		error(E,P1,Y,k3,cmdLineArgs.P);
			
		/*Backprpagation Phase*/ 		
		mtm(dWhy,H,E,cmdLineArgs.M+1,k3,cmdLineArgs.P);
		delta(Why,dWhy,cmdLineArgs.M+1,cmdLineArgs.P,learningrate);
		mmt(H,Why,E,k3,cmdLineArgs.M+1,cmdLineArgs.P);		
		gradient_func(Zh,H,k3,cmdLineArgs.M);		
		mtm(dWxh,X,Zh,cmdLineArgs.N+1,k3,cmdLineArgs.M);
		delta(Wxh,dWxh,cmdLineArgs.N+1,cmdLineArgs.M,learningrate);

	}	
   }

  stop_timer();
  double time = elapsed_time();
  printf( "Time: %lf\n",time);
/*---------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------Print outputs----------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------*/
   if(cmdLineArgs.V)
   {
	/*Need the following 2 statements for Testing*/
	displayMatrix ("input/hidden weights", Wxh, cmdLineArgs.N+1, cmdLineArgs.M);
	displayMatrix ("hidden/output weights", Why, cmdLineArgs.M+1, cmdLineArgs.P);
	/* Useful for analyzing the accuracy of prediction */
	/*if(k3)
	{	
		displayVector ("last input", &X[k3-1][1], cmdLineArgs.N);
		displayVector ("last output", Y[k3-1], cmdLineArgs.P);
		displayVector ("predicted output",P1[k3-1], cmdLineArgs.P);
	}
	else
	{
		displayVector ("last input", &X[b-1][1], cmdLineArgs.N);
		displayVector ("last output", Y[b-1], cmdLineArgs.P);
		displayVector ("predicted output",P1[b-1], cmdLineArgs.P);
	}*/
   }
/*---------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------Free Memory------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------*/
free(inputs);
free(outputs);
free(X);
free(Zh);
free(Zy);
free(H);
free(E);
free(P);
free(P1);
free(sum);
free(Wxh);
free(Why);
free(dWxh);
free(dWhy);
/*-------------------------------------------------------END-----------------------------------------------------*/
return 0;
}


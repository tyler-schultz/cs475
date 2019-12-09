/******************************************************************************
 * util.h
 *
 ******************************************************************************/
#include <getopt.h>
#include <ctype.h>
#include <stdbool.h>

void displayVector (const char *label, double *m, int rows) {
	printf ("\n%s:\n", label);	
	for (int i=0; i<rows; i++) 
		printf ("%10.5lf\t", m[i]);
	printf ("\n");
}

void displayMatrix (const char *label, double **m, int rows, int cols) 
{
 printf ("\n%s:\n", label);
 for(int i = 0; i < rows; ++i )
 {
 	for(int j = 0; j < cols; ++j )
  		printf("%10.5lf\t",m[i][j]);
	printf ("\n");
  }
}

void displayMatrix1 (const char *label, double *m, int rows, int cols)
{
 printf ("\n%s:\n", label);
  for(int i = 0; i < rows; ++i )
   {
       for(int j = 0; j < cols; ++j )
	           printf("%10.5lf\t",m[(i*cols)+j]);
			       printf ("\n");
				     }
					 }

typedef struct 
{
	long N; //Number of input layer neurons
	long M; //Number of hidden layer neurons
	long P; //Number of output layer neurons

	long sample_total; //Number of training samples
	long sample_per_iter; //Number of training samples per iteration 

	long iter; //number of iterations to train 
	
	long numblocks;  //number of thread blocks
	long numthreads; // number of threads per block

	bool V; //verbose

} Params;

//==== parse int abstraction from strtol
int parseInt( char *string )
{
   return (int) strtol( string, NULL, 10 );
}

int parseCmdLineArgs(Params *cmdLineArgs, int argc, char* argv[]){

  //Default values
  cmdLineArgs->N = 1;
  cmdLineArgs->M = 1;
  cmdLineArgs->P = 1;

  cmdLineArgs->sample_total    = 1;
  cmdLineArgs->sample_per_iter = 1;

  cmdLineArgs->iter = 10; 
  
  cmdLineArgs->numblocks = 1;
  cmdLineArgs->numthreads = 1;

  cmdLineArgs->V = false;
  
  // process incoming
  char c;
  
while ((c = getopt (argc, argv, "N:M:P:S:I:B:T:L:VHn:m:p:s:i:b:t:l:hv")) != -1){
    c=toupper(c);
    switch( c ) {

    case 'N': //Number of input layer neurons
      cmdLineArgs->N = parseInt( optarg );
      if(cmdLineArgs->N <= 0)
	{
        fprintf(stderr, "The Number of input layer neurons must be greater than 0: %ld\n",cmdLineArgs->N);
          exit( -1 );
      	}
      break;

      case 'M': //Number of hidden layer neurons
        cmdLineArgs->M = parseInt( optarg );
        if(cmdLineArgs->M <= 0)
	{
        fprintf(stderr, "The Number of hidden layer neurons must be greater than 0: %ld\n",cmdLineArgs->M);
          exit( -1 );
      	}
        break;

     case 'P': //Number of output layer neurons
        cmdLineArgs->P = parseInt( optarg );
         if(cmdLineArgs->P <= 0)
	{
        fprintf(stderr, "The Number of output layer neurons must be greater than 0: %ld\n",cmdLineArgs->P);
          exit( -1 );
      	}
        break;

     case 'S': //Number of training samples
        cmdLineArgs->sample_total = parseInt( optarg );
         if(cmdLineArgs->sample_total <= 0)
	{
        fprintf(stderr, "The Number of training samples must be greater than 0: %ld\n",cmdLineArgs->sample_total);
          exit( -1 );
      	}
        break;
 
     case 'I': //Number of samples per iteration
        cmdLineArgs->sample_per_iter = parseInt( optarg );
         if(cmdLineArgs->sample_per_iter > cmdLineArgs->sample_total)
	{
        fprintf(stderr, "The Number of training samples per iteration must be greater than 0 and lesser total number of samples: %ld\n",cmdLineArgs->sample_per_iter);
          exit( -1 );
      	}
        break;

    case 'B': //Number of thread blocks
        cmdLineArgs->numblocks = parseInt( optarg );
         if(cmdLineArgs->numblocks <= 0)
	{
        fprintf(stderr, "The Number of thread blocks must be greater than 0: %ld\n",cmdLineArgs->numblocks);
          exit( -1 );
      	}
        break;

    case 'T': //Number of threads per blocks
        cmdLineArgs->numthreads = parseInt( optarg );
         if(cmdLineArgs->numthreads <= 0)
	{
        fprintf(stderr, "The Number of thread per blocks must be greater than 0: %ld\n",cmdLineArgs->numthreads);
          exit( -1 );
      	}
        break;

    case 'L': //Number of iterations
        cmdLineArgs->iter = parseInt( optarg );
         if(cmdLineArgs->iter <= 0)
	{
        fprintf(stderr, "The Number of iterations must be greater than 0: %ld\n",cmdLineArgs->iter);
          exit( -1 );
      	}
        break;

     case 'H': // help
        printf("usage: %s\n\t\t"
                  "-N Number of input layer neurons\n\t\t"
		  "-M Number of hidden layer neurons\n\t\t"
		  "-P Number of output layer neurons\n\t\t"
		  "-S Number of training samples\n\t\t"
		  "-I Number of training samples per bunch\n\t\t"
		 /* "-B Number of thread blocks\n\t\t"
		  "-T Number of thread per blocks\n\t\t"*/
		  "-L Number of iterations of time loop\n\t\t"
		  "-V Verbose ON\n\t\t"	
                  "-H usage help, this dialogue\n", argv[0]);
           exit(0);

     case 'V': // verbose;
         cmdLineArgs->V = true;
         break;    
     
      case '?':
         if (optopt == 'N'){
            fprintf (stderr,
                   "Option -%c requires positive int argument: number of input layer neurons.\n",
                   optopt);
          }else if (optopt == 'M'){
            fprintf (stderr,
                     "Option -%c requires positive int argument: number of hidden layer neurons.\n",
                      optopt);
          }else if (optopt == 'P'){
            fprintf (stderr,
                     "Option -%c requires int argument: number of output layer neurons .\n",
                      optopt);
          }else if (optopt == 'S'){
            fprintf (stderr,
                "Option -%c requires int argument: number of samples.\n",
                 optopt);
          }else if (optopt == 'I'){
            fprintf (stderr,
                "Option -%c requires int argument: number of samples per bunch.\n",
                 optopt);
          }else if (optopt == 'L'){
            fprintf (stderr,
                "Option -%c requires int argument: number of iterations of item loop.\n",
                 optopt);
          }else{
            fprintf(stderr, "Unknown option character `\\x%x'.\n", optopt);
          }
          exit(-1);
               
      default:
         exit(0);
    }
  } 
  
  return 0;
}

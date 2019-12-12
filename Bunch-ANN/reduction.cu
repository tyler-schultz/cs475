__global__ void reduction(double** X, double** Y, long A, long B)
{
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(j==0){
       X[i]=0; 
    }
    X[i] += Y[i][j];
    
            
           
}
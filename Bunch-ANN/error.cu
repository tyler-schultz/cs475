__global__ void eroor(double** X, double** Y, double** Z, long B, long C)
{
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i < A && j < B){
          X[i][j] = Y[i][j]-Z[i][j];
    }
            
           
}
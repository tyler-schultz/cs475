__global__ void gradient_func(double** X, double** Y, long A, long B)
{
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i < A && j < B){
        X[i][j] = Y[i][j+1]*(1 - pow (tanh (X[i][j]), 2));   
    }
            
           
}
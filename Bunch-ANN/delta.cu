__global__ void delta(double** Z, double** Y, long A, long B, double C)
{
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i < A && j < B){
        Z(i,j) -= C*Y(i,j);   
    }
            
           
}

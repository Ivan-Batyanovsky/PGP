#include <iostream>
#include <vector>
#include <limits>

#define THREADS_PER_BLOCK 512
typedef double myType;

__global__ void mykernel(myType * v1, myType * v2, myType * res, unsigned long long edge)
{
    if (index < edge)
        res[index] = v1[index] - v2[index];
}

int main()
{
    unsigned long long n;

    myType *d_v1, *d_v2, *d_res;

    std::cin >> n;

    int size = n * (sizeof(myType));

    myType * v1 = new myType[n];
    myType * v2 = new myType[n];
    myType * res = new myType[n];

    for (unsigned long long i = 0; i < n; ++i)
    {
        std::cin >> v1[i];
    }
    for (unsigned long long i = 0; i < n; ++i)
    {
        std::cin >> v2[i];
    }

    cudaMalloc((void **) &d_v1, size);
    cudaMalloc((void **) &d_v2, size);
    cudaMalloc((void **) &d_res, size);

//    std::cout << "Hello world0\n";
    cudaMemcpy(d_v1, v1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v2, v2, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, res, size, cudaMemcpyHostToDevice);

    mykernel<<<(n / THREADS_PER_BLOCK + 1), THREADS_PER_BLOCK>>>(d_v1, d_v2, d_res, n);

    cudaMemcpy(res, d_res, size, cudaMemcpyDeviceToHost);

    std::cout.precision(10);
    std::cout << std::scientific;
    for (long int i = 0; i < n; i++)
    {
       std::cout << res[i] << ' ';
    } std::cout << std::endl;

    delete [] v1;
    delete [] v2;
    delete [] res;

    cudaFree(d_v1);
    cudaFree(d_v2);
    cudaFree(d_res);

    return 0;
}
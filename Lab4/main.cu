#include <thrust/extrema.h>
#include <thrust/device_vector.h>

#include <iostream>

typedef long int li;

#define HANDLE_ERROR(err)                             \
    do { if (err != cudaSuccess) { printf("ERROR: %s\n", cudaGetErrorString(err)); exit(0);} } while (0)


struct comparator
{
    __host__ __device__ bool operator()(double a, double b)
    {
        return std::fabs(a) < std::fabs(b);
    }
};

__global__ void swapRows(double* device_data, long int ind, long int i, long int n)
{
    long int idx = threadIdx.x + blockIdx.x * blockDim.x;
    long int offSet = gridDim.x * blockDim.x;

    double temp;
    for (unsigned long int j = idx; j < n + 1; j += offSet)
    {
        temp = device_data[j * n + i];
        device_data[j * n + i] = device_data[j * n + ind];
        device_data[j * n + ind] = temp;
    }
}

__global__ void maxDiv(double* device_data, long int ind, long int i, long int n)
{
    long int idx = threadIdx.x + blockIdx.x * blockDim.x;
    long int offSet = gridDim.x * blockDim.x;

    double maxElement = 1 / device_data[i * n + ind];
    for (unsigned long int j = idx + i+1; j < n + 1; j += offSet)
    {
        device_data[j * n + i] *= maxElement;
    }
}

__global__ void forward(double* data, long int i, long int n) {

	long int idx = blockIdx.x * blockDim.x + threadIdx.x;
	long int idy = blockIdx.y * blockDim.y + threadIdx.y;
	long int offsetX = gridDim.x * blockDim.x;
	long int offsetY = gridDim.y * blockDim.y;

	for (unsigned long int j = idx + i + 1; j < n; j += offsetX) {
		for (unsigned long int k = idy + i + 1; k < n + 1; k += offsetY) {
			data[j + k * n] -= data[j + i * n] * data[i + k * n] / data[i + i * n];
		}
	}
}
int main(void) {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    long int n;
    std::cin >> n;
    long int nSquare = n * n;
    double *host_data = new double[nSquare + n];
    for (unsigned long int i = 0; i < n; i++) {
        for (unsigned long int j = 0; j < n; j++) {
            std::cin >> host_data[j * n + i];
        }
    }

    for (unsigned long int i = 0; i < n; i++) {
        std::cin >> host_data[nSquare + i];
    }

//    for (int i = 0; i < n + 1; i++)
//    {
//        for (int j = 0; j < n; j++)
//        {
//            std::cout << host_data[i * n + j] << ' ';
//        }
//        std::cout << std::endl;
//    }
//
//    for (int i = 0; i < nSquare + n; i++)
//    {
//        std::cout << host_data[i] << ' ';
//    } std::cout << std::endl;


    // Memory allocation and initialization
    double *device_data;
    cudaMalloc((void **) &device_data, (nSquare + n) * sizeof(double));
    cudaMemcpy(device_data, host_data, (nSquare + n) * sizeof(double), cudaMemcpyHostToDevice);

    // Pointers for max_elem with thrust
    comparator comp;

    thrust::device_ptr<double> begin_p, max_p;

    for (long int i = 0, ind; i < n - 1; i++) {
        // Finding max element
        begin_p = thrust::device_pointer_cast(device_data + i * n);
        max_p = thrust::max_element(begin_p + i, begin_p + n, comp);
        ind = max_p - begin_p;
//        std::cout << "ind: " << ind << ", i:" << i << ", max:" << max_p << std::endl;
        if (ind != i) {
//            std::cout << "swaping rows: " << i << ' ' << ind << std::endl;
            swapRows<<<512, 512>>>(device_data, ind, i, n);
            ind = i;
        }
//        maxDiv<<<512, 512>>>(device_data, ind, i, n);
//        forward<<<dim3(64, 64), dim3(32, 32)>>>(device_data, i, n);
        forward<<<dim3(32, 32), dim3(32, 32)>>>(device_data, i, n);
    }

//    maxDiv<<<512,512>>>(device_data, n - 1, n - 1, n);
//    forward<<<dim3(32, 32), dim3(32, 32)>>>(device_data, n - 1, n - 1);

//    for (int i = n - 1; i > 0; i--)
//    {
//        backward<<<256,256>>>(device_data, i, n);
//    }
    cudaMemcpy(host_data, device_data, (nSquare + n) * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(device_data);

//    host_data[nSquare + n - 1] /= host_data[nSquare - 1];
//    std::cout << "fffffffffffffffffffffffffff\n";
//    std::cout << std::endl;
//    for (int i = 0; i < n + 1; i++)
//    {
//        for (int j = 0; j < n; j++)
//        {
//            std::cout << host_data[i * n + j] << ' ';
//        }
//        std::cout << std::endl;
//    }
//    std::cout << "fffffffffffffffffffffffffff\n";
    for (long int j = n - 1; j >= 0; j--) {
        double sum = host_data[nSquare + j];

        for (long int i = (n + 1) * j + n, tempCounter = 1; i < nSquare; i += n, tempCounter++) {
//            std::cout << "sum: " << sum << "right from save " << host_data[i] << ' ' << host_data[nSquare + j + tempCounter] << std::endl;
            sum -= host_data[i] * host_data[nSquare + j + tempCounter];
        }

        host_data[nSquare + j] = sum / host_data[j * n + j];
    }
//    std::cout << "fffffffffffffffffffffffffff\n";

    std::cout.precision(10);
    std::cout.setf(std::ios::scientific);

    for (unsigned long int i = nSquare, cap = i + n; i < cap; i++) {
        std::cout << host_data[i] << ' ';
    }

    delete [] host_data;
}

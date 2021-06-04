#include <iostream>
#include <string>
#include <fstream>
#include <stdio.h>
#define THREADS_NUMBER 16

texture<uchar4, 2, cudaReadModeElementType> texRef;

__device__ float colourParser(const uchar4 & pixel)
{
    int R = pixel.x,
        G = pixel.y,
        B = pixel.z;

    float Y = (0.299f * R + 0.587f * G + 0.114f * B);

    return Y;
}

__global__ void RobertsMethod(uchar4 * output, int width, int height)
{
    // Calculate normalized texture coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float Gx, Gy;
    unsigned char temp;

    if (x < width && y < height) {
        Gx = colourParser(tex2D(texRef, x + 1, y)) - colourParser(tex2D(texRef, x, y + 1)),
        Gy = colourParser(tex2D(texRef, x, y)) - colourParser(tex2D(texRef, x + 1, y + 1));

    }

//    __syncthreads();

    if (x < width && y < height)
    {
        int grad = sqrtf(Gx * Gx + Gy * Gy);
        temp = grad < 256? grad: 255;
//        temp = (int) sqrtf(Gx * Gx + Gy * Gy);
//        printf("|veavfe|(%d) %f , %f, %d\n", tex2D(texRef, x + 1, y).x, Gx, Gy, temp);
    }

//    __syncthreads();

    if (x < width && y < height)
    {
        output[y * width + x].x = temp;
        output[y * width + x].y = temp;
        output[y * width + x].z = temp;
        output[y * width + x].w = 0;
    }

}


// Host code
int main()
{
    std::string input;
    std::string output;

    std::cin >> input;
    std::cin >> output;

    std::ifstream fin(input, std::ios::binary | std::ios::in);

    int m, n;
    fin.read((char *) &n, 4);
    fin.read((char *) &m, 4);

    size_t size = m * n * sizeof(uchar4);

    uchar4 * h_data = new uchar4[m * n];

    fin.read((char *) h_data, size);

    fin.close();

    cudaArray *arr;
    cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
    cudaMallocArray(&arr, &ch, n, m);
    cudaMemcpyToArray(arr, 0, 0, h_data, size, cudaMemcpyHostToDevice);

    texRef.addressMode[0] = cudaAddressModeClamp;
    texRef.addressMode[1] = cudaAddressModeClamp;
    texRef.channelDesc = ch;
    texRef.filterMode = cudaFilterModePoint;
    texRef.normalized = false;

    cudaBindTextureToArray(texRef, arr, ch);

    uchar4 * dev_data;
    cudaMalloc((void **) &dev_data, size);
    RobertsMethod<<<dim3(512, 512), dim3(32, 32)>>>(dev_data, n, m);

    cudaMemcpy(h_data, dev_data, size, cudaMemcpyDeviceToHost);

    std::ofstream fout(output, std::ios::binary | std::ios::out);

    fout.write((char *) &n, 4);
    fout.write((char *) &m, 4);

    fout.write((char *) h_data, size);
    fout.close();

    cudaUnbindTexture(texRef);
    cudaFreeArray(arr);
    cudaFree(dev_data);

    delete [] h_data;

    return 0;
}

#include <iostream>
#include <string>
#include <fstream>
#include <stdio.h>
using namespace std;

typedef uchar4 ImageType;
typedef double4 ClastersPos;
typedef double DistanceType;

__constant__ int QUANTITY;
__constant__ ClastersPos POSITIONS[32];

void setZero(ClastersPos * pos, int clastersNum) {
    for (int i = 0; i < clastersNum; i++) {
        pos[i].x = 0;
        pos[i].y = 0;
        pos[i].z = 0;
        pos[i].w = 0;
    }
}

__device__ double dist(ImageType f, ClastersPos u)
{
    return sqrtf((f.x - u.x) * (f.x - u.x) + (f.y - u.y) * (f.y - u.y) + (f.z - u.z) * (f.z - u.z));
}

bool hasChanged(const ClastersPos * const old, const ClastersPos * const nw, const int n)
{
    for (int i = 0; i < n; i++)
    {
        if (old[i].x != nw[i].x || old[i].y != nw[i].y || old[i].z != nw[i].z || old[i].w != nw[i].w)
            return true;
    }
    return false;
}

__global__ void kernel(ImageType *data, int n, int m)
{
    // Calculate normalized texture coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int xOffset = blockDim.x * gridDim.x;
    int yOffset = blockDim.y * gridDim.y;

    for (int i = y; i < m; i += yOffset)
    {
        for (int j = x; j < n; j += xOffset)
        {
            ImageType threadValue = data[i * n + j];
            DistanceType minDist = 1.7976931348623158e+308;
            DistanceType tempDist;

            int tempClaster;

            for (int claster = 0; claster < QUANTITY; claster++)
            {
                tempDist = dist(threadValue, POSITIONS[claster]);

                if (tempDist < minDist)
                {
                    minDist = tempDist;
                    tempClaster = claster;
                }
            }

            data[i * n + j].w = tempClaster;
        }
    }
}

int main()
{
    int width, height;
    string input, output;
    cin >> input;
    cin >> output;

    ifstream fin(input, std::ios::binary | std::ios::in);

    fin.read((char *) &width, 4);
    fin.read((char *) &height, 4);

    ImageType *host_data = new ImageType [width * height];
    fin.read((char *) host_data, sizeof(ImageType) * height * width);

    fin.close();


    int clastersNum;
    cin >> clastersNum;

    ClastersPos * host_positions = new ClastersPos [clastersNum];
    for (int i = 0, x, y; i < clastersNum; i++)
    {
        cin >> x >> y;

        host_positions[i].x = host_data[y * width + x].x;
        host_positions[i].y = host_data[y * width + x].y;
        host_positions[i].z = host_data[y * width + x].z;
    }

    ImageType * device_data;
    cudaMalloc((void **) &device_data, sizeof(ImageType) * height * width);
    cudaMemcpy(device_data, host_data, sizeof(ImageType) * height * width, cudaMemcpyHostToDevice);

    ClastersPos * device_positions;
    cudaMalloc((void **) &device_positions, sizeof(ClastersPos) * clastersNum);

    cudaMemcpyToSymbol(QUANTITY, &clastersNum, sizeof(int));

    ClastersPos * host_output = new ClastersPos [clastersNum];

    while (true)
    {
        cudaMemcpyToSymbol(POSITIONS, host_positions, clastersNum * sizeof(ClastersPos));

        kernel<<<dim3(16,16), dim3(16,16)>>>(device_data, width, height);

        cudaMemcpy(host_data, device_data, sizeof(ImageType) * height * width, cudaMemcpyDeviceToHost);

//        cudaDeviceSynchronize();

        setZero(host_output, clastersNum);

        for (int i = 0; i < height; i++)
            for (int j = 0, claster, ind; j < width; j++)
            {
                claster = host_data[i * width + j].w;
                ind = i * width + j;
                host_output[claster].x += host_data[ind].x;
                host_output[claster].y += host_data[ind].y;
                host_output[claster].z += host_data[ind].z;
                host_output[claster].w += 1;
            }

        for (int i = 0; i < clastersNum; i++)
        {
            host_output[i].x /= host_output[i].w;
            host_output[i].y /= host_output[i].w;
            host_output[i].z /= host_output[i].w;
        }

        if (!hasChanged(host_positions, host_output, clastersNum)) break;

        for (int i = 0; i < clastersNum; i++)
        {
            host_positions[i] = host_output[i];
        }
    }
//    printClastersPos(host_data, width, height);

    std::ofstream fout(output, std::ios::binary | std::ios::out);

    fout.write((char *) &width, 4);
    fout.write((char *) &height, 4);

    fout.write((char *) host_data, sizeof(ImageType) * height * width);
    fout.close();

    return 0;
}
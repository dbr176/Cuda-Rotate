#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

#define CHECK(value) {                                          \
    cudaError_t _m_cudaStat = value;                                        \
    if (_m_cudaStat != cudaSuccess) {                                       \
        cout<< "Error:" << cudaGetErrorString(_m_cudaStat) \
            << " at line " << __LINE__ << " in file " << __FILE__ << "\n"; \
        exit(1);                                                            \
    } }


#define TILE_WIDTH 32
#define BLOCK_ROWS 8

__global__ void rotate(int *a, int *c, int rows, int cols)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;

    int aInd = (cols - x) * cols + y;
    int cInd = y * rows + x;

    if (aInd < rows * cols)
        c[cInd] = a[aInd];
}

__global__ void rotateOpt(int *a, int *c, int rows, int cols)
{
    __shared__ int tile[TILE_WIDTH][TILE_WIDTH + 1];

    int x = threadIdx.x + blockIdx.x * TILE_WIDTH;
    int y = threadIdx.y + blockIdx.y * TILE_WIDTH;

    int aInd = y * cols + x;
    if (aInd < rows * cols)
        tile[threadIdx.y][threadIdx.x] = a[aInd];

    __syncthreads();

    x = threadIdx.x + blockIdx.y * TILE_WIDTH;
    y = threadIdx.y + blockIdx.x * TILE_WIDTH;

    int cInd = y * rows + (cols - x);
    if (cInd < cols * rows)
        c[cInd] = tile[threadIdx.x][threadIdx.y];
}

int main(void)
{
    Mat image;
    image = imread("pic1.png", -1);   // Read the file
    if (!image.data)                              // Check for invalid input
    {
        cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    int N = image.rows * image.cols;
    int *host_a, *host_c_check;
    int *dev_a, *dev_c;

    cudaEvent_t startCUDA, stopCUDA;
    clock_t startCPU;
    float elapsedTimeCUDA, elapsedTimeCPU;


    host_a = (int*)image.data;

    host_c_check = new int[N];
    startCPU = clock();
    //#pragma omp parallel for
    for (int x = 0; x < image.rows; x++)
        for (int y = 0; y < image.cols; y++)
            host_c_check[y * image.rows + x] = host_a[x * image.cols + y];

    elapsedTimeCPU = (double)(clock() - startCPU) / CLOCKS_PER_SEC;
    cout << "CPU sum time = " << elapsedTimeCPU * 1000 << " ms\n";
    cout << "CPU memory throughput = " << N * sizeof(int) / elapsedTimeCPU / 1024 / 1024 / 1024 << " Gb/s\n";

    cudaEventCreate(&startCUDA);
    cudaEventCreate(&stopCUDA);


    CHECK(cudaMalloc(&dev_a, N * sizeof(int)));
    CHECK(cudaMalloc(&dev_c, N * sizeof(int)));
    CHECK(cudaMemset(dev_c, 0, N * sizeof(int)));
    CHECK(cudaMemcpy(dev_a, host_a, N * sizeof(int), cudaMemcpyHostToDevice));

    cudaEventRecord(startCUDA, 0);

    // (N + 511) / 512, 512
    //rotate<<<dim3((image.rows + 31) / 32, (image.cols + 31) / 32), dim3(32, 32, 1) >>>
    //  (dev_a, dev_c, image.rows, image.cols);

    dim3 dimGrid((image.cols + TILE_WIDTH - 1) / TILE_WIDTH,
        (image.rows + TILE_WIDTH - 1) / TILE_WIDTH,
        1
    );

    // BLOCK_ROWS
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    rotateOpt<<<dimGrid, dimBlock >>>(dev_a, dev_c, image.rows, image.cols);

    cudaEventRecord(stopCUDA, 0);
    cudaEventSynchronize(stopCUDA);
    CHECK(cudaGetLastError());

    cudaEventElapsedTime(&elapsedTimeCUDA, startCUDA, stopCUDA);

    cout << "CUDA sum time = " << elapsedTimeCUDA << " ms\n";
    cout << "CUDA memory throughput = " << N * sizeof(int) / elapsedTimeCUDA / 1024 / 1024 / 1.024 << " Gb/s\n";

    Mat img2(image.cols, image.rows, image.type());

    //for(int i = 0; i < N; i++)
    //  host_a[i] = 0;
    CHECK(cudaMemcpy(img2.data, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));

    CHECK(cudaFree(dev_a));
    CHECK(cudaFree(dev_c));

    imwrite("pic2.png", img2);

    return 0;
}


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <chrono>

using namespace std;

float makeCPU(int* inData, int N);

float cudaParallel(int* inData, int N);

void init(int* inData, int N)
{
	for (int i = 0; i < N; i++)
		inData[i] = 100 - i + 1;
}

__global__ void reductionKernelMinimum(int *inData, int N)
{
	int tId = threadIdx.x;
	int k = blockIdx.x * blockDim.x + threadIdx.x;

	int before = k;
	int after = (k + 1);

	while (after < N)
	{
		if (inData[before] < inData[after])
			continue;
		else
			inData[before] = inData[after];

		before *= 2;
		after *= 2;

		if (before >= N)
			break;

		__syncthreads();
	}
}

int main()
{
	int N;
	while (true)
	{
		cout << "Enter number of elements: " << endl;
		cin >> N;
		const int elementsCount = N;

		cout << "Reduction for: " << elementsCount << endl;

		int *a = new int[elementsCount];

		init(a, elementsCount);
		float gpuTime = cudaParallel(a, elementsCount);
		cout << "Time on gpu: " << gpuTime << endl;
		
		init(a, elementsCount);
		float cpuTime = makeCPU(a, elementsCount);
		cout << "Time on cpu in ns " << cpuTime << endl;
 	}

	return 0;
}

float makeCPU(int* inData, int N)
{
	int min = inData[0];

	std::chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();
	for (int i = 1; i < N; i++) 
	{
		if (inData[i] < min) 
			min = inData[i];
	}
	end = std::chrono::system_clock::now();

	int elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>
		(end - start).count();

	return elapsed;
}

float cudaParallel(int* inData, int N)
{
	int* deviceData;

	cudaMalloc((void**)&deviceData, N * sizeof(int));
	cudaMemcpy(deviceData, inData, N * sizeof(int), cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	float gpuTime = 0.0f;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);


	dim3 threads(256, 1, 1);
	dim3 blocks(N / 256, 1);

	cudaEventRecord(start, 0);

	reductionKernelMinimum <<<blocks, threads>>> (deviceData, N);

	cudaEventRecord(stop, 0);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpuTime, start, stop);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaMemcpy(inData, deviceData, N * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(deviceData);

	return gpuTime;
}
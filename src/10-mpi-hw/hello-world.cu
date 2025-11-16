#include "../cuda-util.h"

#include <mpi.h>


int main(int argc, char *argv[]) {
    // initialize MPI
    MPI_Init(&argc, &argv);

    // get rank information
    int numRanks = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // choose GPU
    int numDevicesPerNode = 0;
    checkCudaError(cudaGetDeviceCount(&numDevicesPerNode));

    int deviceId = rank % numDevicesPerNode;
    checkCudaError(cudaSetDevice(deviceId));

    std::cout << "Rank " << rank << " using device " << deviceId << std::endl;

    // wait for all ranks to reach this point
    MPI_Barrier(MPI_COMM_WORLD);

    // compute the sum of all rank ids from device data
    int *d_rank;
    checkCudaError(cudaMalloc(&d_rank, sizeof(int)));
    checkCudaError(cudaMemcpy(d_rank, &rank, sizeof(int), cudaMemcpyHostToDevice));

    int *d_rankSum;
    checkCudaError(cudaMalloc(&d_rankSum, sizeof(int)));
    checkCudaError(cudaMemset(d_rankSum, 0, sizeof(int)));

    MPI_Reduce(
        d_rank,         // send 'buffer' - device pointer
        d_rankSum,      // receive 'buffer' - device pointer
        1,              // number of elements
        MPI_INT,        // data type of elements
        MPI_SUM,        // reduce operation
        0,              // root rank - this rank will receive the result
        MPI_COMM_WORLD  // communicator - all ranks participate
    );

    if (0 == rank) {    // rankSum is only valid at root rank
        int rankSum = 0;
        checkCudaError(cudaMemcpy(&rankSum, d_rankSum, sizeof(int), cudaMemcpyDeviceToHost));

        std::cout << "Sum of all ranks is: " << rankSum << std::endl;
    }

    checkCudaError(cudaFree(d_rank));
    checkCudaError(cudaFree(d_rankSum));

    // finalize MPI
    MPI_Finalize();

    return 0;
}

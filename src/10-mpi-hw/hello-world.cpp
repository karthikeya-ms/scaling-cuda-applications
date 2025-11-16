#include "../util.h"

#include <mpi.h>


int main(int argc, char *argv[]) {
    // initialize MPI
    MPI_Init(&argc, &argv);

    // get rank information
    int numRanks = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // hello world from each rank
    std::cout << "Hello world from rank " << rank << " out of " << numRanks << std::endl;

    // wait for all ranks to reach this point
    MPI_Barrier(MPI_COMM_WORLD);

    // compute the sum of all rank ids

    int rankSum = 0;
    MPI_Reduce(
        &rank,          // send 'buffer'
        &rankSum,       // receive 'buffer'
        1,              // number of elements
        MPI_INT,        // data type of elements
        MPI_SUM,        // reduce operation
        0,              // root rank - this rank will receive the result
        MPI_COMM_WORLD  // communicator - all ranks participate
    );

    if (0 == rank)      // rankSum is only valid at root rank
        std::cout << "Sum of all ranks is: " << rankSum << std::endl;

    // wait for all ranks to reach this point
    MPI_Barrier(MPI_COMM_WORLD);

    if (numRanks < 2) {
        if (0 == rank) 
            std::cout << "The next examples require at least 2 ranks" << std::endl;

        MPI_Finalize();

        return 0;
    }

    // demonstrate point-to-point communication
    if (0 == rank) {
        const char msg[] = "Greetings from rank 0!";
        MPI_Send(
            msg,                // message buffer
            sizeof(msg),        // number of elements to send
            MPI_CHAR,           // data type of elements
            1,                  // destination rank
            0,                  // message tag
            MPI_COMM_WORLD      // communicator
        );
    } else if (1 == rank) {
        char msg[256];
        MPI_Recv(
            msg,                // message buffer
            sizeof(msg),        // number of elements to receive
            MPI_CHAR,           // data type of elements
            0,                  // source rank
            0,                  // message tag
            MPI_COMM_WORLD,     // communicator
            MPI_STATUS_IGNORE   // ignore the reported status
        );
        std::cout << "Rank 1 received message: " << msg << std::endl;
    }

    // wait for all ranks to reach this point
    MPI_Barrier(MPI_COMM_WORLD);

    // demonstrate asynchronous point-to-point communication
    if (0 == rank) {
        const char msg[] = "Asynchronous greetings from rank 0!";
        MPI_Request request;
        MPI_Isend(
            msg,                // message buffer
            sizeof(msg),        // number of elements to send
            MPI_CHAR,           // data type of elements
            1,                  // destination rank
            0,                  // message tag
            MPI_COMM_WORLD,     // communicator
            &request            // request required for waiting later
        );

        // ... some asynchronous work ...

        MPI_Wait(&request, MPI_STATUS_IGNORE); // wait for send to complete
    } else if (1 == rank) {
        char msg[256];
        MPI_Request request;
        MPI_Irecv(
            msg,                // message buffer
            sizeof(msg),        // number of elements to receive (maximum number of elements)
            MPI_CHAR,           // data type of elements
            0,                  // source rank
            0,                  // message tag
            MPI_COMM_WORLD,     // communicator
            &request            // request required for waiting later
        );

        // ... some asynchronous work ...

        MPI_Wait(&request, MPI_STATUS_IGNORE); // wait for receive to complete
        std::cout << "Rank 1 received asynchronous message: " << msg << std::endl;
    }

    // finalize MPI
    MPI_Finalize();

    return 0;
}

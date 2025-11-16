#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>


#ifdef __NVCC__
#   define FCT_DECORATOR __host__ __device__
#else
#   define FCT_DECORATOR
#endif


void parseCLA_2d(int argc, char* const* argv, size_t& globalNumCellsX, size_t& globalNumCellsY, size_t& numItWarmUp, size_t& numItTimed, size_t& printInterval) {
    // default values
    globalNumCellsX = 1024;
    globalNumCellsY = globalNumCellsX;
    numItWarmUp = 2;
    numItTimed = 10;
    printInterval = 0;

    // override with command line arguments
    int i = 1;
    if (argc > i) globalNumCellsX = atoi(argv[i]);
    ++i;
    if (argc > i) globalNumCellsY = atoi(argv[i]);
    ++i;
    if (argc > i) numItWarmUp = atoi(argv[i]);
    ++i;
    if (argc > i) numItTimed = atoi(argv[i]);
    ++i;
    if (argc > i) printInterval = atoi(argv[i]);
    ++i;
}

void printStats(const std::chrono::duration<double> elapsedSeconds, size_t numItTimed, size_t nCells, size_t numBytesPerCell, size_t numFlopsPerCell) {
    std::cout << "  #cells / #it:  " << nCells << " / " << numItTimed << "\n";
    std::cout << "  elapsed time:  " << 1e3 * elapsedSeconds.count() << " ms\n";
    std::cout << "  per iteration: " << 1e3 * elapsedSeconds.count() / numItTimed << " ms\n";
    std::cout << "  MLUP/s:        " << 1e-6 * nCells * numItTimed / elapsedSeconds.count() << "\n";
    std::cout << "  bandwidth:     " << 1e-9 * numBytesPerCell * nCells * numItTimed / elapsedSeconds.count() << " GB/s\n";
    std::cout << "  compute:       " << 1e-9 * numFlopsPerCell * nCells * numItTimed / elapsedSeconds.count() << " GFLOP/s\n";
}

FCT_DECORATOR size_t ceilingDivide(size_t a, size_t b) {
    return (a + b - 1) / b;
}

FCT_DECORATOR size_t ceilToMultipleOf(size_t a, size_t b) {
    return ceilingDivide(a, b) * b;
}

#pragma once

#include "file-util.h"


constexpr double alpha = 0.2;


inline void initTemperature(double *__restrict__ u, size_t globalNumCellsX, size_t globalNumCellsY) {
    memset(u, 0, globalNumCellsX * globalNumCellsY * sizeof(double));

    // inject pulse
    u[globalNumCellsX / 3 + (globalNumCellsY / 3) * globalNumCellsX] = 1.0;
}


inline void initTemperature(double *__restrict__ u, double *__restrict__ uNew, size_t globalNumCellsX, size_t globalNumCellsY) {
    memset(u, 0, globalNumCellsX * globalNumCellsY * sizeof(double));
    memset(uNew, 0, globalNumCellsX * globalNumCellsY * sizeof(double));

    // inject pulse
    u[globalNumCellsX / 3 + (globalNumCellsY / 3) * globalNumCellsX] = 1.0;
}


inline void initTemperaturePatch(double *__restrict__ u, size_t globalNumCellsX, size_t globalNumCellsY, size_t localNumCellsX, size_t localNumCellsY, size_t offsetX, size_t offsetY) {
    memset(u, 0, localNumCellsX * localNumCellsY * sizeof(double));

    // compute local and global pulse position
    auto globalX = globalNumCellsX / 3;
    auto globalY = globalNumCellsY * 2 / 3;
    auto localX = globalX - offsetX;
    auto localY = globalY - offsetY;
    
    // inject pulse if its position is included in the current patch
    if (globalX >= offsetX && globalX < offsetX + localNumCellsX &&
        globalY >= offsetY && globalY < offsetY + localNumCellsY)
        u[localX + localY * localNumCellsX] = 1.0;
}


inline void initTemperaturePatch(double *__restrict__ u, double *__restrict__ uNew, size_t globalNumCellsX, size_t globalNumCellsY, size_t localNumCellsX, size_t localNumCellsY, size_t offsetX, size_t offsetY) {
    memset(uNew, 0, localNumCellsX * localNumCellsY * sizeof(double));
    initTemperaturePatch(u, globalNumCellsX, globalNumCellsY, localNumCellsX, localNumCellsY, offsetX, offsetY);
}


inline double accumulateTemperature(const double *__restrict__ u, size_t globalNumCellsX, size_t globalNumCellsY) {
    double acc = 0;
    for (size_t i1 = 1; i1 < globalNumCellsY - 1; ++i1) {
        for (size_t i0 = 1; i0 < globalNumCellsX - 1; ++i0) {
            acc += u[i0 + i1 * globalNumCellsX];
        }
    }

    return acc;
}

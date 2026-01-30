/*---------------------------------------------------------------------------*\
|                                                                             |
| cudaLBM: CUDA-based moment representation Lattice Boltzmann Method          |
| Developed at UDESC - State University of Santa Catarina                     |
| Website: https://www.udesc.br                                               |
| Github: https://github.com/geoenergiaUDESC/cudaLBM                          |
|                                                                             |
\*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*\

Copyright (C) 2023 UDESC Geoenergia Lab
Authors: Nathan Duggins (Geoenergia Lab, UDESC)

This implementation is derived from concepts and algorithms developed in:
  MR-LBM: Moment Representation Lattice Boltzmann Method
  Copyright (C) 2021 CERNN
  Developed at Universidade Federal do Paraná (UFPR)
  Original authors: V. M. de Oliveira, M. A. de Souza, R. F. de Souza
  GitHub: https://github.com/CERNN/MR-LBM
  Licensed under GNU General Public License version 2

License
    This file is part of cudaLBM.

    cudaLBM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

Description
    Function definitions and includes specific to the fieldCalculate executable

Namespace
    LBM

SourceFiles
    testExecutable.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_TESTEXECUTABLE_CUH
#define __MBLBM_TESTEXECUTABLE_CUH

#include "../../src/LBMIncludes.cuh"
#include "../../src/LBMTypedefs.cuh"
#include "../../src/streaming/streaming.cuh"
#include "../../src/collision/collision.cuh"
#include "../../src/blockHalo/blockHalo.cuh"
#include "../../src/fileIO/fileIO.cuh"
#include "../../src/runTimeIO/runTimeIO.cuh"
#include "../../src/functionObjects/objectRegistry.cuh"
#include "../../src/array/array.cuh"
#include "../../src/boundaryConditions/boundaryConditions.cuh"

namespace LBM
{

#ifdef JETFLOW
    using BoundaryConditions = jetFlow;
    __device__ __host__ [[nodiscard]] inline consteval bool periodicX() noexcept { return true; }
    __device__ __host__ [[nodiscard]] inline consteval bool periodicY() noexcept { return true; }
#endif

#ifdef LIDDRIVENCAVITY
    using BoundaryConditions = lidDrivenCavity;
    __device__ __host__ [[nodiscard]] inline consteval bool periodicX() noexcept { return false; }
    __device__ __host__ [[nodiscard]] inline consteval bool periodicY() noexcept { return false; }
#endif

    using VelocitySet = D3Q19;
    using Collision = secondOrder;
    using BlockHalo = device::halo<VelocitySet, periodicX(), periodicY()>;

    __device__ __host__ [[nodiscard]] inline consteval label_t smem_alloc_size() noexcept { return 0; }

    __host__ [[nodiscard]] inline consteval label_t MIN_BLOCKS_PER_MP() noexcept { return 2; }
#define launchBoundsD3Q19 __launch_bounds__(block::maxThreads(), MIN_BLOCKS_PER_MP())

    template <typename T>
    __device__ inline void print(const T deviceID, const T bx, const T GLOBAL_X_BLOCK_OFFSET, const T by, const T GLOBAL_Y_BLOCK_OFFSET, const T bz, const T GLOBAL_Z_BLOCK_OFFSET) noexcept
    {
        if constexpr (sizeof(T) == 8)
        {
            printf("deviceID: %lu\n{\n    blockIdx {%lu, %lu, %lu};\n};\n\n", deviceID, bx + GLOBAL_X_BLOCK_OFFSET, by + GLOBAL_Y_BLOCK_OFFSET, bz + GLOBAL_Z_BLOCK_OFFSET);
        }
        else
        {
            printf("deviceID: %u\n{\n    blockIdx {%u, %u, %u};\n};\n\n", deviceID, bx + GLOBAL_X_BLOCK_OFFSET, by + GLOBAL_Y_BLOCK_OFFSET, bz + GLOBAL_Z_BLOCK_OFFSET);
        }
    }

    launchBoundsD3Q19 __global__ void testKernel(
        label_t *const ptrRestrict deviceIDPtr,
        const label_t NUM_BLOCK_X,
        const label_t NUM_BLOCK_Y,
        const label_t GLOBAL_X_BLOCK_OFFSET,
        const label_t GLOBAL_Y_BLOCK_OFFSET,
        const label_t GLOBAL_Z_BLOCK_OFFSET,
        const label_t correctDevice)
    {
        // Always a multiple of 32, so no need to check this(I think)
        if constexpr (out_of_bounds_check())
        {
            if (device::out_of_bounds())
            {
                return;
            }
        }

        const label_t tx = threadIdx.x;
        const label_t ty = threadIdx.y;
        const label_t tz = threadIdx.z;
        const label_t bx = blockIdx.x;
        const label_t by = blockIdx.y;
        const label_t bz = blockIdx.z;

        const label_t idx = (tx + block::nx<label_t>() * (ty + block::ny<label_t>() * (tz + block::nz<label_t>() * (bx + NUM_BLOCK_X * (by + NUM_BLOCK_Y * bz)))));

        const label_t deviceID = deviceIDPtr[idx];

        if (!(deviceID == correctDevice))
        {
            printf("Bad deviceID\n");
        }

        if ((threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0))
        {
            print(deviceID, bx, GLOBAL_X_BLOCK_OFFSET, by, GLOBAL_Y_BLOCK_OFFSET, bz, GLOBAL_Z_BLOCK_OFFSET);
        }

        deviceIDPtr[idx] = deviceID + 100;
    }
}
#endif
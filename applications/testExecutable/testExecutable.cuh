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
    using BoundaryConditions = typename boundaryConditions::traits<boundaryConditions::caseName()>::type;
    using VelocitySet = D3Q27;
    using Collision = secondOrder;
    using BlockHalo = device::halo<VelocitySet, BoundaryConditions::periodicX(), BoundaryConditions::periodicY(), BoundaryConditions::periodicZ()>;

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
        const device::threadCoordinate Tx;

        const device::blockCoordinate Bx;

        const device::pointCoordinate point(Tx, Bx);

        // Always a multiple of 32, so no need to check this(I think)
        if constexpr (out_of_bounds_check())
        {
            if (device::out_of_bounds(point))
            {
                return;
            }
        }

        const label_t idx = device::idx(Tx, Bx);

        if ((threadIdx.x == 7) && (threadIdx.y == 7) && (threadIdx.z == 7))
        {
            printf("Accessing idx %lu\n", static_cast<uint64_t>(idx));
        }

        const label_t deviceID = deviceIDPtr[idx];

        // return;

        if (!(deviceID == correctDevice))
        {
            printf("Bad deviceID\n");
        }

        if ((threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0))
        {
            print(deviceID, Bx.value<axis::X>(), GLOBAL_X_BLOCK_OFFSET, Bx.value<axis::Y>(), GLOBAL_Y_BLOCK_OFFSET, Bx.value<axis::Z>(), GLOBAL_Z_BLOCK_OFFSET);
        }

        deviceIDPtr[idx] = deviceID + 100;
    }
}
#endif
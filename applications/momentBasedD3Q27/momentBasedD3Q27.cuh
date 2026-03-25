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
    Aliases and kernel definitions for the thermal D3Q27 moment representation
    lattice Boltzmann model

Namespace
    LBM

SourceFiles
    momentBasedD3Q27.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_MOMENTBASEDD3Q27_CUH
#define __MBLBM_MOMENTBASEDD3Q27_CUH

#include "../../src/LBMIncludes.cuh"
#include "../../src/typedefs/typedefs.cuh"
#include "../../src/streaming/streaming.cuh"
#include "../../src/collision/collision.cuh"
#include "../../src/blockHalo/blockHalo.cuh"
#include "../../src/fileIO/fileIO.cuh"
#include "../../src/runTimeIO/runTimeIO.cuh"
#include "../../src/functionObjects/objectRegistry.cuh"
#include "../../src/array/array.cuh"
#include "../../src/boundaryConditions/boundaryConditions.cuh"
#include "../../src/momentBasedLBM/kernel.cuh"

namespace LBM
{
    using BoundaryConditions = boundaryConditions::traits<boundaryConditions::caseName()>::type;
    using VelocitySet = D3Q27<Thermal>;
    using Collision = secondOrder;
    using BlockHalo = device::halo<VelocitySet, BoundaryConditions::periodicX(), BoundaryConditions::periodicY(), BoundaryConditions::periodicZ()>;

#ifndef launchBoundsD3Q27
#define launchBoundsD3Q27 __launch_bounds__(block::maxThreads(), MIN_BLOCKS_PER_MP())
#endif

    /**
     * @brief Implements solution of the lattice Boltzmann method using the moment representation and the D3Q27 velocity set
     * @param[in] devPtrs Collection of 10 pointers to device arrays on the GPU
     * @param[in] readBuffer Collection of read-only pointers to the block halo faces used during streaming
     * @param[in] writeBuffer Collection of mutable pointers to the block halo faces used after streaming
     **/
    launchBoundsD3Q27 __global__ void momentBasedD3Q27(
        const device::ptrCollection<10, scalar_t> devPtrs,
        const device::ptrCollection<6, const scalar_t> readBuffer,
        const device::ptrCollection<6, scalar_t> writeBuffer)
    {
        extern __shared__ scalar_t shared_buffer[];

        momentBasedLBM<BoundaryConditions, VelocitySet, Collision, BlockHalo>(devPtrs, readBuffer, writeBuffer, shared_buffer);
    }
}

#endif
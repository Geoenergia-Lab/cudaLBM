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
#include "../../src/typedefs/typedefs.cuh"
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
    using BoundaryConditions = boundaryConditions::traits<boundaryConditions::caseName()>::type;
    using VelocitySet = D3Q27;
    using Collision = secondOrder;
    using BlockHalo = device::halo<VelocitySet, BoundaryConditions::periodicX(), BoundaryConditions::periodicY(), BoundaryConditions::periodicZ()>;

    __device__ __host__ [[nodiscard]] inline consteval device::label_t smem_alloc_size() noexcept
    {
        if constexpr (std::is_same<VelocitySet, D3Q27>::value)
        {
            return block::sharedMemoryBufferSize<VelocitySet, NUMBER_MOMENTS<host::label_t>()>(sizeof(scalar_t));
        }
        else
        {
            return 0;
        }
    }

    __host__ [[nodiscard]] inline consteval device::label_t MIN_BLOCKS_PER_MP() noexcept { return 2; }

#ifndef launchBoundsD3Q27
#define launchBoundsD3Q27 __launch_bounds__(block::maxThreads(), MIN_BLOCKS_PER_MP())
#endif

    // /**
    //  * @brief Implements solution of the lattice Boltzmann method using the moment representation and the D3Q19 velocity set
    //  * @param[in] devPtrs Collection of 10 pointers to device arrays on the GPU
    //  * @param[in] blockHalo Object containing pointers to the block halo faces used to exchange the population densities
    //  **/
    // launchBoundsD3Q27 __global__ void momentBasedD3Q19(
    //     const device::ptrCollection<10, scalar_t> devPtrs,
    //     const device::ptrCollection<6, const scalar_t> readBuffer,
    //     const device::ptrCollection<6, scalar_t> writeBuffer)
    // {
    //     const thread::coordinate Tx;

    //     const block::coordinate Bx;

    //     const device::pointCoordinate point(Tx, Bx);

    //     // Index into global arrays
    //     const device::label_t idx = device::idx(Tx, Bx);

    //     // Into block arrays
    //     const device::label_t tid = block::idx(Tx);

    //     // Always a multiple of 32, so no need to check this(I think)
    //     if constexpr (out_of_bounds_check())
    //     {
    //         if (device::out_of_bounds(point))
    //         {
    //             return;
    //         }
    //     }

    //     // Prefetch devPtrs into L2
    //     // device::constexpr_for<0, NUMBER_MOMENTS()>(
    //     //     [&](const auto moment)
    //     //     {
    //     //         cache::prefetch<cache::Level::L2, cache::Policy::evict_last>(&(devPtrs.ptr<moment>()[idx]));
    //     //     });

    //     // Declare shared memory (flattened)
    //     __shared__ thread::array<scalar_t, block::sharedMemoryBufferSize<VelocitySet, NUMBER_MOMENTS<host::label_t>()>()> shared_buffer;

    //     // Coalesced read from global memory
    //     thread::array<scalar_t, NUMBER_MOMENTS()> moments;
    //     device::constexpr_for<0, NUMBER_MOMENTS()>(
    //         [&](const auto moment)
    //         {
    //             const device::label_t ID = tid * m_i<NUMBER_MOMENTS() + 1>() + m_i<moment>();
    //             shared_buffer[ID] = devPtrs.ptr<moment>()[idx];
    //             if constexpr (moment == index::rho)
    //             {
    //                 moments[moment] = shared_buffer[ID] + rho0();
    //             }
    //             else
    //             {
    //                 moments[moment] = shared_buffer[ID];
    //             }
    //         });

    //     block::sync();

    //     // Reconstruct the population from the moments
    //     thread::array<scalar_t, VelocitySet::Q()> pop = VelocitySet::reconstruct(moments);

    //     // Save populations in shared memory
    //     streaming::save<VelocitySet>(pop, shared_buffer, tid);

    //     block::sync();

    //     // Pull from shared memory
    //     streaming::pull<VelocitySet>(pop, shared_buffer, Tx);

    //     // Pull pop from global memory in cover nodes
    //     BlockHalo::pull(pop, readBuffer, Tx, Bx, point);

    //     block::sync();

    //     if constexpr (std::is_same<BoundaryConditions, lidDrivenCavity>::value)
    //     {
    //         // Calculate the moments either at the boundary or interior
    //         {
    //             const normalVector boundaryNormal(point);

    //             if (boundaryNormal.isBoundary())
    //             {
    //                 BoundaryConditions::calculate_moments<VelocitySet>(pop, moments, boundaryNormal, shared_buffer.data(), Tx, point);
    //             }
    //             else
    //             {
    //                 velocitySet::calculate_moments<VelocitySet>(pop, moments);
    //             }
    //         }
    //     }

    //     if constexpr (std::is_same<BoundaryConditions, jetFlow>::value)
    //     {
    //         // Compute post-stream moments
    //         velocitySet::calculate_moments<VelocitySet>(pop, moments);
    //         {
    //             // Update the shared buffer with the refreshed moments
    //             device::constexpr_for<0, NUMBER_MOMENTS()>(
    //                 [&](const auto moment)
    //                 {
    //                     const device::label_t ID = tid * label_constant<NUMBER_MOMENTS() + 1>() + label_constant<moment>();
    //                     shared_buffer[ID] = moments[moment];
    //                 });
    //         }

    //         block::sync();

    //         // Calculate the moments at the boundary
    //         {
    //             const normalVector boundaryNormal(point);

    //             if (boundaryNormal.isBoundary())
    //             {
    //                 BoundaryConditions::calculate_moments<VelocitySet>(pop, moments, boundaryNormal, shared_buffer.data(), Tx, point);
    //             }
    //         }
    //     }

    //     // Scale the moments correctly
    //     velocitySet::scale(moments);

    //     // Collide
    //     Collision::collide(moments);

    //     // Coalesced write to global memory
    //     device::constexpr_for<0, NUMBER_MOMENTS()>(
    //         [&](const auto moment)
    //         {
    //             if constexpr (moment == index::rho)
    //             {
    //                 devPtrs.ptr<moment>()[idx] = moments[moment] - rho0();
    //             }
    //             else
    //             {
    //                 devPtrs.ptr<moment>()[idx] = moments[moment];
    //             }
    //         });

    //     // Save the populations to the block halo
    //     // BlockHalo::save_from_shared(shared_buffer, writeBuffer);
    //     BlockHalo::save(pop, moments, writeBuffer, Tx, Bx, point);
    // }

    launchBoundsD3Q27 __global__ void momentBasedD3Q27(
        const device::ptrCollection<10, scalar_t> devPtrs,
        const device::ptrCollection<6, const scalar_t> readBuffer,
        const device::ptrCollection<6, scalar_t> writeBuffer)
    {
        const thread::coordinate Tx;

        // const normalVector blockNormal(Tx);

        const block::coordinate Bx;

        const device::pointCoordinate point(Tx, Bx);

        // Index into global arrays
        const device::label_t idx = device::idx(Tx, Bx);

        // Into block arrays
        const device::label_t tid = block::idx(Tx);

        // Always a multiple of 32, so no need to check this(I think)
        if constexpr (out_of_bounds_check())
        {
            if (device::out_of_bounds(point))
            {
                return;
            }
        }

        // Prefetch devPtrs into L2
        device::constexpr_for<0, NUMBER_MOMENTS()>(
            [&](const auto moment)
            {
                cache::prefetch<cache::Level::L2, cache::Policy::evict_last>(&(devPtrs.ptr<moment>()[idx]));
            });

        // Declare shared memory (flattened)
        extern __shared__ scalar_t shared_buffer[];

        // Coalesced read from global memory
        thread::array<scalar_t, NUMBER_MOMENTS()> moments;
        device::constexpr_for<0, NUMBER_MOMENTS()>(
            [&](const auto moment)
            {
                const device::label_t ID = tid * m_i<NUMBER_MOMENTS() + 1>() + m_i<moment>();
                shared_buffer[ID] = devPtrs.ptr<moment>()[idx];
                if constexpr (moment == index::rho)
                {
                    moments[moment] = shared_buffer[ID] + rho0();
                }
                else
                {
                    moments[moment] = shared_buffer[ID];
                }
            });

        block::sync();

        // Reconstruct the population from the moments
        thread::array<scalar_t, VelocitySet::Q()>
            pop = VelocitySet::reconstruct(moments);

        // Save/pull from shared memory
        {
            // Save populations in shared memory
            streaming::save<VelocitySet>(pop, shared_buffer, tid);

            block::sync();

            // Pull from shared memory
            streaming::pull<VelocitySet>(pop, shared_buffer, Tx);

            // Pull pop from global memory in cover nodes
            BlockHalo::pull(pop, readBuffer, Tx, Bx, point);

            block::sync();
        }

        if constexpr (std::is_same<BoundaryConditions, lidDrivenCavity>::value)
        {
            // Calculate the moments either at the boundary or interior
            {
                const normalVector boundaryNormal(point);

                if (boundaryNormal.isBoundary())
                {
                    BoundaryConditions::calculate_moments<VelocitySet>(pop, moments, boundaryNormal, shared_buffer, Tx, point);
                }
                else
                {
                    velocitySet::calculate_moments<VelocitySet>(pop, moments);
                }
            }
        }

        if constexpr (std::is_same<BoundaryConditions, jetFlow>::value)
        {
            // Compute post-stream moments
            velocitySet::calculate_moments<VelocitySet>(pop, moments);
            {
                // Update the shared buffer with the refreshed moments
                device::constexpr_for<0, NUMBER_MOMENTS()>(
                    [&](const auto moment)
                    {
                        const device::label_t ID = tid * label_constant<NUMBER_MOMENTS() + 1>() + label_constant<moment>();
                        shared_buffer[ID] = moments[moment];
                    });
            }

            block::sync();

            // Calculate the moments at the boundary
            {
                const normalVector boundaryNormal(point);

                if (boundaryNormal.isBoundary())
                {
                    BoundaryConditions::calculate_moments<VelocitySet>(pop, moments, boundaryNormal, shared_buffer, Tx, point);
                }
            }
        }

        // Scale the moments correctly
        velocitySet::scale(moments);

        // Collide
        Collision::collide(moments);

        // Coalesced write to global memory
        device::constexpr_for<0, NUMBER_MOMENTS()>(
            [&](const auto moment)
            {
                if constexpr (moment == index::rho)
                {
                    devPtrs.ptr<moment>()[idx] = moments[moment] - rho0();
                }
                else
                {
                    devPtrs.ptr<moment>()[idx] = moments[moment];
                }
            });

        // Save the populations to the block halo
        // BlockHalo::save_from_shared(shared_buffer, writeBuffer);
        BlockHalo::save(pop, moments, writeBuffer, Tx, Bx, point);
    }

    template <typename T>
    __host__ [[nodiscard]] inline constexpr auto kernel() noexcept;

    // template <>
    // __host__ [[nodiscard]] inline constexpr auto kernel<D3Q19>() noexcept
    // {
    //     return momentBasedD3Q19;
    // }

    template <>
    __host__ [[nodiscard]] inline constexpr auto kernel<D3Q27>() noexcept
    {
        return momentBasedD3Q27;
    }
}

#endif
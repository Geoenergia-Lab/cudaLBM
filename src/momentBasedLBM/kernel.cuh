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
    Definition of the main GPU kernel

Namespace
    LBM::host, LBM::device

SourceFiles
    momentBasedLBM.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_MOMENTBASEDLBM_CUH
#define __MBLBM_MOMENTBASEDLBM_CUH

namespace LBM
{
    /**
     * @brief Determines the amount of shared memory required for a kernel based on the velocity set
     **/
    template <class VelocitySet>
    __device__ __host__ [[nodiscard]] inline consteval device::label_t smem_alloc_size() noexcept
    {
        if constexpr (true)
        {
            return block::sharedMemoryBufferSize<VelocitySet, NUMBER_MOMENTS<host::label_t>()>(sizeof(scalar_t));
        }
        else
        {
            return 0;
        };
    }

    /**
     * @brief Minimum number of blocks per streaming microprocessor
     **/
    __host__ [[nodiscard]] inline consteval device::label_t MIN_BLOCKS_PER_MP() noexcept { return 1; }

    /**
     * @brief Implements solution of the lattice Boltzmann method using the moment representation and a chosen velocity set
     * @tparam BoundaryConditions The boundary conditions of the solver
     * @tparam VelocitySet The velocity set to use for streaming
     * @tparam Collision The collision model
     * @tparam BlockHalo The class handling inter-block streaming
     * @param[in] devPtrs Collection of 10 pointers to device arrays on the GPU
     * @param[in] readBuffer Collection of read-only pointers to the block halo faces used during streaming
     * @param[in] writeBuffer Collection of mutable pointers to the block halo faces used after streaming
     * @param[in] sharedBuffer Inline or externally stored shared memory buffer
     **/
    template <class BoundaryConditions, class VelocitySet, class Collision, class BlockHalo, class SharedBuffer>
    __device__ inline void momentBasedLBM(
        const device::ptrCollection<10, scalar_t> &devPtrs,
        const device::ptrCollection<6, const scalar_t> &readBuffer,
        const device::ptrCollection<6, scalar_t> &writeBuffer,
        SharedBuffer &sharedBuffer)
    {
        static_assert(std::is_same_v<BlockHalo, device::halo<VelocitySet, BoundaryConditions::periodicX(), BoundaryConditions::periodicY(), BoundaryConditions::periodicZ()>>);

        const thread::coordinate Tx;

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
        device::constexpr_for<0, NUMBER_MOMENTS<false>()>(
            [&](const auto moment)
            {
                cache::prefetch<cache::Level::L2, cache::Policy::evict_last>(&(devPtrs.ptr<moment>()[idx]));
            });

        // Coalesced read from global memory
        thread::array<scalar_t, NUMBER_MOMENTS<false>()> moments;
        device::constexpr_for<0, NUMBER_MOMENTS<false>()>(
            [&](const auto moment)
            {
                const device::label_t ID = tid * m_i<NUMBER_MOMENTS<false>() + 1>() + m_i<moment>();
                sharedBuffer[ID] = devPtrs.ptr<moment>()[idx];
                if constexpr (moment == index::rho)
                {
                    moments[moment] = sharedBuffer[ID] + rho0();
                }
                else
                {
                    moments[moment] = sharedBuffer[ID];
                }
            });

        block::sync();

        // Reconstruct the population from the moments
        thread::array<scalar_t, VelocitySet::Q()> pop = VelocitySet::reconstruct(moments);

        // Save/pull from shared memory
        {
            // Save populations in shared memory
            streaming::save<VelocitySet>(pop, sharedBuffer, tid);

            block::sync();

            // Pull from shared memory
            streaming::pull<VelocitySet>(pop, sharedBuffer, Tx);

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
                    BoundaryConditions::template calculate_moments<VelocitySet>(pop, moments, boundaryNormal, sharedBuffer, Tx, point);
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
                device::constexpr_for<0, NUMBER_MOMENTS<false>()>(
                    [&](const auto moment)
                    {
                        const device::label_t ID = tid * label_constant<NUMBER_MOMENTS<false>() + 1>() + label_constant<moment>();
                        sharedBuffer[ID] = moments[moment];
                    });
            }

            block::sync();

            // Calculate the moments at the boundary
            {
                const normalVector boundaryNormal(point);

                if (boundaryNormal.isBoundary())
                {
                    BoundaryConditions::template calculate_moments<VelocitySet>(pop, moments, boundaryNormal, sharedBuffer, Tx, point);
                }
            }
        }

        // Scale the moments correctly
        velocitySet::scale(moments);

        // Collide
        Collision::collide(moments);

        // Coalesced write to global memory
        device::constexpr_for<0, NUMBER_MOMENTS<false>()>(
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
        BlockHalo::save(pop, moments, writeBuffer, Tx, Bx, point);
    }
}

#endif
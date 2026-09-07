/*---------------------------------------------------------------------------*\
|                                                                             |
| HermiteLBM: CUDA-based moment representation Lattice Boltzmann Method       |
| Developed at UDESC - State University of Santa Catarina                     |
| Website: https://www.udesc.br                                               |
| Github: https://github.com/Geoenergia-Lab/HermiteLBM                        |
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
    This file is part of HermiteLBM.

    HermiteLBM is free software: you can redistribute it and/or modify it
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
    kernel.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_MOMENTBASEDLBM_KERNEL_CUH
#define __MBLBM_MOMENTBASEDLBM_KERNEL_CUH

namespace LBM
{
    template <class VelocitySet, class BoundaryConditions, class Collision>
    struct momentBasedLBMKernel
    {
        /**
         * @brief Alias for the block halo
         **/
        using BlockHalo = device::halo<VelocitySet, BoundaryConditions>;
        using Streaming = streaming<VelocitySet>;

        /**
         * @brief Saves a momentsArray object to its original pointers
         * @param[out] devPtrs The pointers to save to
         * @param[in] moments The array of 10 moments
         * @param[in] idx The index into the global array
         **/
        template <const host::label_t i>
        __device__ static inline constexpr void saveToPtr(
            const device::ptrColl_t &devPtrs,
            const momentsArray &moments,
            const device::label_t idx) noexcept
        {
            if constexpr (i == axis::index<axis::NO_DIRECTION>())
            {
                devPtrs.ptr<i>()[idx] = moments[i] - rho0();
            }
            else
            {
                devPtrs.ptr<i>()[idx] = moments[i];
            }
        }

        /**
         * @brief Implements solution of the lattice Boltzmann method using the moment representation and a chosen velocity set
         * @tparam BoundaryConditions The boundary conditions of the solver
         * @tparam Collision The collision model
         * @tparam BlockHalo The class handling inter-block streaming
         * @param[in] devPtrs Collection of 10 pointers to device arrays on the GPU
         * @param[in] readBuffer Collection of read-only pointers to the block halo faces used during streaming
         * @param[in] writeBuffer Collection of mutable pointers to the block halo faces used after streaming
         * @param[in] sharedBuffer Inline or externally stored shared memory buffer
         * @param[in] Tx Three-dimensional thread coordinates
         * @param[in] Bx Three-dimensional block coordinates
         **/
        template <class SharedBuffer>
        __device__ static inline void momentBasedLBM(
            const device::ptrColl_t &devPtrs,
            const device::ptrCollection<6, const scalar_t> &readBuffer,
            const device::ptrCollection<6, scalar_t> &writeBuffer,
            SharedBuffer &sharedBuffer,
            const thread::coordinate &Tx,
            const block::coordinate &Bx) noexcept
        {
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

            // Coalesced read from global memory
            momentsArray moments{
                devPtrs.ptr<axis::index<axis::NO_DIRECTION>()>()[idx] + rho0(),
                devPtrs.ptr<axis::index<axis::X>()>()[idx],
                devPtrs.ptr<axis::index<axis::Y>()>()[idx],
                devPtrs.ptr<axis::index<axis::Z>()>()[idx],
                devPtrs.ptr<axis::index<axis::X, axis::X>()>()[idx],
                devPtrs.ptr<axis::index<axis::X, axis::Y>()>()[idx],
                devPtrs.ptr<axis::index<axis::X, axis::Z>()>()[idx],
                devPtrs.ptr<axis::index<axis::Y, axis::Y>()>()[idx],
                devPtrs.ptr<axis::index<axis::Y, axis::Z>()>()[idx],
                devPtrs.ptr<axis::index<axis::Z, axis::Z>()>()[idx]};
            block::sync();

            // Reconstruct the population from the moments
            thread::array<scalar_t, VelocitySet::Q()> pop;
            VelocitySet::reconstruct(pop, moments);

            // Save populations in shared memory
            Streaming::save(pop, sharedBuffer, tid);
            block::sync();

            // Pull from shared memory
            Streaming::pull(pop, sharedBuffer, Tx);

            // Pull pop from global memory in cover nodes
            BlockHalo::pull(pop, readBuffer, Tx, Bx, point);
            block::sync();

            // Update the post-streaming moments according to the interior and/or boundary conditions
            if constexpr (BoundaryConditions::appliesCondition())
            {
                BoundaryConditions::template calculate_moments<VelocitySet>(pop, moments, sharedBuffer, Tx, point, tid);
            }
            else
            {
                VelocitySet::template calculate_moments(moments, pop);
            }

            // Scale the moments correctly
            velocitySetBase::scale(moments);

            // Collide
            Collision::collide(moments);

            // Coalesced write to global memory
            device::constexpr_for<0, NUMBER_MOMENTS()>(
                [&](const auto moment)
                {
                    saveToPtr<moment>(devPtrs, moments, idx);
                });

            // Save the populations to the block halo
            if constexpr (use_cooperative_halo())
            {
                VelocitySet::reconstruct<false>(pop, moments);
                BlockHalo::transpose_to_shared(pop, writeBuffer, sharedBuffer, Tx, Bx, point);
                BlockHalo::save_from_shared(sharedBuffer, writeBuffer, Tx, Bx);
            }
            else
            {
                BlockHalo::save(pop, moments, writeBuffer, Tx, Bx, point);
            }
        }

        /**
         * @overload Wraps the implementation, calculating an offset block ID for multi-GPU compatibility
         * @param[in] bzOffset Offset to the block ID in the Z axis
         **/
        template <class SharedBuffer>
        __device__ static inline void momentBasedLBM(
            const device::ptrColl_t &devPtrs,
            const device::ptrCollection<6, const scalar_t> &readBuffer,
            const device::ptrCollection<6, scalar_t> &writeBuffer,
            SharedBuffer &sharedBuffer,
            const device::label_t bzOffset) noexcept
        {
            static_assert(std::is_same_v<BlockHalo, device::halo<VelocitySet, BoundaryConditions>>);

            const thread::coordinate Tx;

            const block::coordinate Bx(blockIdx.x, blockIdx.y, blockIdx.z + bzOffset);

            momentBasedLBM(devPtrs, readBuffer, writeBuffer, sharedBuffer, Tx, Bx);
        }

        /**
         * @overload Wraps the implementation for a single GPU system
         **/
        template <class SharedBuffer>
        __device__ static inline void momentBasedLBM(
            const device::ptrColl_t &devPtrs,
            const device::ptrCollection<6, const scalar_t> &readBuffer,
            const device::ptrCollection<6, scalar_t> &writeBuffer,
            SharedBuffer &sharedBuffer) noexcept
        {
            static_assert(std::is_same_v<BlockHalo, device::halo<VelocitySet, BoundaryConditions>>);

            const thread::coordinate Tx;

            const block::coordinate Bx;

            momentBasedLBM(devPtrs, readBuffer, writeBuffer, sharedBuffer, Tx, Bx);
        }
    };

    namespace kernel
    {
        /**
         * @brief Implements solution of the lattice Boltzmann method using the moment representation and the D3Q19 velocity set
         * @param[in] devPtrs Collection of 10 pointers to device arrays on the GPU
         * @param[in] readBuffer Collection of read-only pointers to the block halo faces used during streaming
         * @param[in] writeBuffer Collection of mutable pointers to the block halo faces used after streaming
         **/
        __launch_bounds__(block::maxThreads(), MIN_BLOCKS_PER_MP<VelocitySet>()) __global__ void momentBasedLBM(
            const device::ptrColl_t devPtrs,
            const device::ptrCollection<6, const scalar_t> readBuffer,
            const device::ptrCollection<6, scalar_t> writeBuffer,
            const device::label_t bzOffset)
        {
            if constexpr (VelocitySet::smem_alloc_size() == 0)
            {
                __shared__ thread::array<scalar_t, block::sharedMemoryBufferSize<VelocitySet::Q(), NUMBER_MOMENTS<host::label_t>()>()> sharedBuffer;

                momentBasedLBMKernel<VelocitySet, BoundaryConditions, Collision>::momentBasedLBM(devPtrs, readBuffer, writeBuffer, sharedBuffer, bzOffset);
            }
            else
            {
                extern __shared__ scalar_t sharedBuffer[];

                momentBasedLBMKernel<VelocitySet, BoundaryConditions, Collision>::momentBasedLBM(devPtrs, readBuffer, writeBuffer, sharedBuffer, bzOffset);
            }
        }

        /**
         * @brief Launches the kernel on a given list of streams
         * @param[in] mesh Lattice mesh object containing information about the grid and block dimensions
         * @param[in] programCtrl Program control object containing information about the devices and streams
         * @param[in] devPtrs Collection of pointers to device arrays on the GPU, used to pass the data to the kernel
         * @param[in] haloPtrs Collection of pointers to the block halo faces used during streaming
         * @param[in] timeStep Current time step of the simulation, used to determine which halo buffers to use for reading and writing
         * @param[in] idxStreams The streams on which to launch the kernel
         * @param[in] bzOffset Offsets to the block ID in the Z axis
         **/
        template <const host::label_t N>
        __host__ void launchHelper(
            const host::latticeMesh &mesh,
            const programControl &programCtrl,
            const ptrCollection &devPtrs,
            const haloBuffer<VelocitySet> &haloPtrs,
            const host::label_t timeStep,
            const std::array<host::label_t, N> &idxStreams,
            const std::array<device::label_t, N> &bzOffsets) noexcept
        {
            // Pre-sync and launch the kernels
            for (host::label_t deviceIdx = 0; deviceIdx < programCtrl.deviceList().size(); deviceIdx++)
            {
                // Set the active device
                errorHandler::handleInline(cudaSetDevice(programCtrl.deviceList()[deviceIdx]));

                // Sync the streams to ensure previous operations are complete before launching new kernels
                for (const host::label_t idxStream : idxStreams)
                {
                    programCtrl.streams().synchronize(device::idxStream(deviceIdx, idxStream));
                }

                // Launch the kernels for the specified streams and block offsets
                for (host::label_t idxStream = 0; idxStream < idxStreams.size(); idxStream++)
                {
                    kernel::launch<momentBasedLBM, VelocitySet::smem_alloc_size()>(
                        mesh.gridBlock()[device::idxStream(deviceIdx, idxStreams[idxStream])],
                        programCtrl.streams()[device::internalStreamID(deviceIdx)],
                        devPtrs[deviceIdx],
                        haloPtrs.readBuffer(deviceIdx, timeStep),
                        haloPtrs.writeBuffer(deviceIdx, timeStep),
                        bzOffsets[idxStream]);
                }
            }

            // Sync the streams
            for (host::label_t deviceIdx = 0; deviceIdx < programCtrl.deviceList().size(); deviceIdx++)
            {
                for (const host::label_t idxStream : idxStreams)
                {
                    programCtrl.streams().synchronize(device::idxStream(deviceIdx, idxStream));
                }
            }
        }

        /**
         * @brief Launches the lattice Boltzmann kernel for all devices and streams, ensuring proper synchronization and device selection
         * @param[in] mesh Lattice mesh object containing information about the grid and block dimensions
         * @param[in] programCtrl Program control object containing information about the devices and streams
         * @param[in] devPtrs Collection of pointers to device arrays on the GPU, used to pass the data to the kernel
         * @param[in] haloPtrs Collection of pointers to the block halo faces used during streaming
         * @param[in] timeStep Current time step of the simulation, used to determine which halo buffers to use for reading and writing
         **/
        __host__ inline void launchInternal(
            const host::latticeMesh &mesh,
            const programControl &programCtrl,
            const ptrCollection &devPtrs,
            const haloBuffer<VelocitySet> &haloPtrs,
            const host::label_t timeStep) noexcept
        {
            constexpr const std::array<host::label_t, 1> idxStreams = {static_cast<device::label_t>(1)};
            constexpr const std::array<device::label_t, 1> bzOffsets = {static_cast<device::label_t>(1)};
            launchHelper(mesh, programCtrl, devPtrs, haloPtrs, timeStep, idxStreams, bzOffsets);
        }

        /**
         * @brief Launches the lattice Boltzmann kernel for all devices and streams, ensuring proper synchronization and device selection
         * @param[in] mesh Lattice mesh object containing information about the grid and block dimensions
         * @param[in] programCtrl Program control object containing information about the devices and streams
         * @param[in] devPtrs Collection of pointers to device arrays on the GPU, used to pass the data to the kernel
         * @param[in] haloPtrs Collection of pointers to the block halo faces used during streaming
         * @param[in] devComm Device communicator object used to handle inter-device communication of halo buffers
         * @param[in] timeStep Current time step of the simulation, used to determine which halo buffers to use for reading and writing
         **/
        __host__ inline void launchBoundary(
            const host::latticeMesh &mesh,
            const programControl &programCtrl,
            const ptrCollection &devPtrs,
            const haloBuffer<VelocitySet> &haloPtrs,
            const deviceCommunicator<VelocitySet> &devComm,
            const host::label_t timeStep) noexcept
        {
            constexpr const std::array<host::label_t, 2> idxStreams = {static_cast<device::label_t>(0), static_cast<device::label_t>(2)};
            const std::array<device::label_t, 2> bzOffsets = {static_cast<device::label_t>(0), static_cast<device::label_t>(mesh.blocksPerDevice<axis::Z>() - static_cast<host::label_t>(1))};
            launchHelper(mesh, programCtrl, devPtrs, haloPtrs, timeStep, idxStreams, bzOffsets);
            devComm.exchange(timeStep);
        }
    }
}

#endif
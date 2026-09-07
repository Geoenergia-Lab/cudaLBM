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
    A class handling the individual faces of the device halo.

Namespace
    LBM::device

SourceFiles
    deviceHalo.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_DEVICEHALO_CUH
#define __MBLBM_DEVICEHALO_CUH

#include "initialisation.cuh"

namespace LBM
{
    /**
     * @brief A class handling the individual faces of the device halo.
     * @tparam VelocitySet The velocity set (D3Q19 or D3Q27)
     **/
    template <class VelocitySet>
    class haloBuffer
    {
    public:
        /**
         * @brief Aliases
         **/
        template <typename T>
        using doubleBuffer = device::ptrCollection<12, T>;
        template <typename T>
        using singleBuffer = device::ptrCollection<6, T>;

        /**
         * @brief Constructor
         * @param[in] rho Device scalar field containing the density values on the GPU
         * @param[in] U Device vector field containing the velocity values on the GPU
         * @param[in] Pi Device symmetric tensor field containing the stress tensor values on the GPU
         * @param[in] mesh Host lattice mesh containing the mesh information on the CPU
         * @param[in] programCtrl Host program control containing the program information on the CPU
         **/
        __host__ [[nodiscard]] haloBuffer(
            const device::scalarField<VelocitySet, time::instantaneous> &rho,
            const device::vectorField<VelocitySet, time::instantaneous> &U,
            const device::symmetricTensorField<VelocitySet, time::instantaneous> &Pi,
            const host::latticeMesh &mesh,
            const programControl &programCtrl)
            : ptrs_(initialise(rho, U, Pi, mesh, programCtrl)) {}

        /**
         * @brief Destructor - Frees the device memory allocated for the block halo buffers
         **/
        __host__ ~haloBuffer()
        {
            for (const doubleBuffer<scalar_t> &collection : ptrs_)
            {
                for (host::label_t i = 0; i < 12; ++i)
                {
                    device::free(collection.ptr(i));
                }
            }
        }

        /**
         * @brief Returns a read-only buffer for the specified device and time step
         * @param[in] deviceIdx The index of the device
         * @param[in] timeStep The time step
         * @return A read-only buffer
         **/
        __host__ [[nodiscard]] inline constexpr const singleBuffer<const scalar_t> readBuffer(
            const host::label_t deviceIdx,
            const host::label_t timeStep) const noexcept
        {
            const host::label_t readPtrOffset = 6 * ((timeStep + 1) % 2);
            const doubleBuffer<scalar_t> &devPtrs = ptrs_[deviceIdx];
            return {
                devPtrs.constPtr(0 + readPtrOffset),
                devPtrs.constPtr(1 + readPtrOffset),
                devPtrs.constPtr(2 + readPtrOffset),
                devPtrs.constPtr(3 + readPtrOffset),
                devPtrs.constPtr(4 + readPtrOffset),
                devPtrs.constPtr(5 + readPtrOffset)};
        }

        /**
         * @brief Returns a mutable buffer for the specified device and time step
         * @param[in] deviceIdx The index of the device
         * @param[in] timeStep The time step
         * @return A mutable buffer
         **/
        __host__ [[nodiscard]] inline constexpr const singleBuffer<scalar_t> writeBuffer(
            const host::label_t deviceIdx,
            const host::label_t timeStep) const noexcept
        {
            const host::label_t writePtrOffset = 6 * (timeStep % 2);
            const doubleBuffer<scalar_t> &devPtrs = ptrs_[deviceIdx];
            return {
                devPtrs.ptr(0 + writePtrOffset),
                devPtrs.ptr(1 + writePtrOffset),
                devPtrs.ptr(2 + writePtrOffset),
                devPtrs.ptr(3 + writePtrOffset),
                devPtrs.ptr(4 + writePtrOffset),
                devPtrs.ptr(5 + writePtrOffset)};
        }

    private:
        /**
         * @brief A collection of pointers to device arrays on the GPU used for the block halo.
         **/
        const std::vector<doubleBuffer<scalar_t>> ptrs_;

        /**
         * @brief Calculates the allocation size for the block halo buffers
         * @param[in] nx The number of blocks in the X direction
         * @param[in] ny The number of blocks in the Y direction
         * @param[in] nz The number of blocks in the Z direction
         * @return The allocation size
         **/
        template <const axis::type alpha>
        __host__ [[nodiscard]] static inline constexpr host::label_t allocSize(
            const host::label_t nx,
            const host::label_t ny,
            const host::label_t nz) noexcept
        {
            return VelocitySet::template QF<host::label_t>() * ((static_cast<host::label_t>(nx) * static_cast<host::label_t>(ny) * static_cast<host::label_t>(nz) * block::nx<host::label_t>() * block::ny<host::label_t>() * block::nz<host::label_t>()) / block::n<alpha, host::label_t>());
        }

        /**
         * @brief Initialises the block halo buffers by launching the initialisation kernel
         ** @param[in] rho Device scalar field containing the density values on the GPU
         * @param[in] U Device vector field containing the velocity values on the GPU
         * @param[in] Pi Device symmetric tensor field containing the stress tensor values on the GPU
         * @param[in] mesh Host lattice mesh containing the mesh information on the CPU
         * @param[in] programCtrl Host program control containing the program information on the CPU
         **/
        __host__ [[nodiscard]] const doubleBuffer<scalar_t> initialise_ptrs(
            const device::scalarField<VelocitySet, time::instantaneous> &rho,
            const device::vectorField<VelocitySet, time::instantaneous> &U,
            const device::symmetricTensorField<VelocitySet, time::instantaneous> &Pi,
            const host::latticeMesh &mesh,
            const programControl &programCtrl,
            const host::label_t deviceIdx) const
        {
            errorHandler::handle(cudaDeviceSynchronize());
            errorHandler::handle(cudaSetDevice(programCtrl.deviceList()[deviceIdx]));
            errorHandler::handle(cudaDeviceSynchronize());

            scalar_t *haloPtrs[12];

            // Allocate the halo buffers for the current device
            {
                // Calculate the allocation size
                const host::label_t nxBlocksTrue = mesh.blocksPerDevice<axis::X>();
                const host::label_t nyBlocksTrue = mesh.blocksPerDevice<axis::Y>();
                const host::label_t nzBlocksTrue = mesh.blocksPerDevice<axis::Z>();
                const host::label_t xAllocationSize = allocSize<axis::X>(nxBlocksTrue, nyBlocksTrue, nzBlocksTrue);
                const host::label_t yAllocationSize = allocSize<axis::Y>(nxBlocksTrue, nyBlocksTrue, nzBlocksTrue);
                const host::label_t zAllocationSize = allocSize<axis::Z>(nxBlocksTrue, nyBlocksTrue, nzBlocksTrue);

                // Allocate
                for (host::label_t N = 0; N < 2; N++)
                {
                    haloPtrs[0 + (6 * N)] = device::allocate<scalar_t>(xAllocationSize);
                    haloPtrs[1 + (6 * N)] = device::allocate<scalar_t>(xAllocationSize);

                    haloPtrs[2 + (6 * N)] = device::allocate<scalar_t>(yAllocationSize);
                    haloPtrs[3 + (6 * N)] = device::allocate<scalar_t>(yAllocationSize);

                    haloPtrs[4 + (6 * N)] = device::allocate<scalar_t>(zAllocationSize);
                    haloPtrs[5 + (6 * N)] = device::allocate<scalar_t>(zAllocationSize);
                }
            }

            // Launch the initialisation kernel
            const doubleBuffer<scalar_t> haloBuffers(
                haloPtrs[0], haloPtrs[1], haloPtrs[2], haloPtrs[3], haloPtrs[4], haloPtrs[5],
                haloPtrs[6], haloPtrs[7], haloPtrs[8], haloPtrs[9], haloPtrs[10], haloPtrs[11]);

            const device::ptrCollection<NUMBER_MOMENTS<host::label_t>(), const scalar_t> devPtrs(
                rho.self().constPtr(deviceIdx),
                U.x().constPtr(deviceIdx),
                U.y().constPtr(deviceIdx),
                U.z().constPtr(deviceIdx),
                Pi.xx().constPtr(deviceIdx),
                Pi.xy().constPtr(deviceIdx),
                Pi.xz().constPtr(deviceIdx),
                Pi.yy().constPtr(deviceIdx),
                Pi.yz().constPtr(deviceIdx),
                Pi.zz().constPtr(deviceIdx));

            if (runTime::program_status.load() == runTime::GOOD)
            {
                kernel::launch<kernel::momentBasedLBMInitialisation<VelocitySet>()>(
                    mesh,
                    programCtrl.streams()[device::internalStreamID(deviceIdx)],
                    devPtrs,
                    haloBuffers);
            }

            errorHandler::handle(cudaDeviceSynchronize());

            return haloBuffers;
        }

        /**
         * @brief Initialises the block halo buffers for all devices by launching the initialisation kernel for each device
         * @param[in] rho Device scalar field containing the density values on the GPU
         * @param[in] U Device vector field containing the velocity values on the GPU
         * @param[in] Pi Device symmetric tensor field containing the stress tensor values on the GPU
         * @param[in] mesh Host lattice mesh containing the mesh information on the CPU
         * @param[in] programCtrl Host program control containing the program information on the CPU
         * @return A vector of double buffers containing the block halo buffers for each device
         **/
        __host__ [[nodiscard]] const std::vector<doubleBuffer<scalar_t>> initialise(
            const device::scalarField<VelocitySet, time::instantaneous> &rho,
            const device::vectorField<VelocitySet, time::instantaneous> &U,
            const device::symmetricTensorField<VelocitySet, time::instantaneous> &Pi,
            const host::latticeMesh &mesh,
            const programControl &programCtrl) const noexcept
        {
            const host::label_t numDevices = programCtrl.deviceList().size();

            std::vector<doubleBuffer<scalar_t>> vec;

            ifAllocationAllowed(
                [&]()
                {
                    vec.reserve(numDevices);

                    for (host::label_t devIdx = 0; devIdx < numDevices; ++devIdx)
                    {
                        vec.emplace_back(initialise_ptrs(rho, U, Pi, mesh, programCtrl, devIdx));
                    }
                });

            return vec;
        }
    };
}

#endif
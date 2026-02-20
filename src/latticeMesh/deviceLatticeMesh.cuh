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
    A class holding information about the solution grid on the GPU(s)

Namespace
    LBM::device

SourceFiles
    deviceLatticeMesh.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_DEVICELATTICEMESH_CUH
#define __MBLBM_DEVICELATTICEMESH_CUH

#ifdef DEVICE_LATTICEMESH_READY

namespace LBM
{
    namespace device
    {
        /**
         * @brief Point indices descriptor
         * @details Stores point indices in 3D space
         **/
        struct pointLabel_t
        {
            /**
             * @brief Constructor for pointLabel_t
             * @param[in] label A dim3 struct containing the point indices
             **/
            __device__ [[nodiscard]] inline constexpr pointLabel_t(const dim3 &label) noexcept
                : x(static_cast<label_t>(label.x)),
                  y(static_cast<label_t>(label.y)),
                  z(static_cast<label_t>(label.z)) {}

            /**
             * @brief Constructor for pointLabel_t
             * @param[in] X The point index in the x-direction
             * @param[in] Y The point index in the y-direction
             * @param[in] Z The point index in the z-direction
             **/
            __device__ [[nodiscard]] inline constexpr pointLabel_t(const label_t X, const label_t Y, const label_t Z) noexcept
                : x(X),
                  y(Y),
                  z(Z) {}

            const label_t x;
            const label_t y;
            const label_t z;
        };

        /**
         * @class latticeMesh
         * @brief Represents the computational grid for LBM simulations
         *
         * This class encapsulates the 3D lattice grid information including
         * dimensions, block decomposition, and physical properties. It handles
         * initialization from configuration files, validation of grid parameters,
         * and synchronization of grid properties with GPU device memory.
         **/
        class latticeMesh
        {
        public:
            __host__ [[nodiscard]] inline latticeMesh(
                const host::latticeMesh &hostMesh,
                const pointLabel_t &deviceID,
                const blockLabel_t &nGPUs) noexcept
                : mesh_(hostMesh),
                  blockOffsets_({deviceID.x / (hostMesh.nxBlocks() / nGPUs.nx), deviceID.y / (hostMesh.nyBlocks() / nGPUs.ny), deviceID.z / (hostMesh.nzBlocks() / nGPUs.nz)}),
                  blockSpan_({hostMesh.nxBlocks() / nGPUs.nx, hostMesh.nyBlocks() / nGPUs.ny, hostMesh.nzBlocks() / nGPUs.nz}),
                  deviceID_(getPartitionDeviceID(blockOffsets_, blockSpan_, nGPUs)),
                  size_((mesh_.nx() / nGPUs.nx) * (mesh_.ny() / nGPUs.ny) * (mesh_.nz() / nGPUs.nz))
            {
                std::cout << "device::latticeMesh[" << deviceID.x << ", " << deviceID.y << ", " << deviceID.z << "]:" << std::endl;
                std::cout << "{" << std::endl;
                std::cout << "    deviceID: " << deviceID_ << std::endl;
                std::cout << "    span(idx): " << size() << std::endl;
                blockOffsets_.print("blockOffset");
                blockSpan_.print("blockSpan");
                std::cout << "};" << std::endl;
            };

            /**
             * @brief Get the size of the memory allocation for this partition of the global mesh for a single field
             **/
            __host__ [[nodiscard]] inline constexpr label_t size() const noexcept
            {
                return size_;
            }

        private:
            /**
             * @brief Const reference to the global lattice mesh
             **/
            const host::latticeMesh &mesh_;

            /**
             * @brief Global block offsets: the blocks at which this partition of the mesh begins
             **/
            const blockLabel_t blockOffsets_;

            /**
             * @brief Device block span: the number of mesh blocks per device
             **/
            const blockLabel_t blockSpan_;

            /**
             * @brief Device index: the flattened index of the device
             **/
            const deviceIndex_t deviceID_;

            const label_t size_;

            /**
             * @brief Computes the device ID assigned to a mesh partition
             * @param[in] blockOffsets The x, y and z block offsets of the device
             * @param[in] blockSpan The number of x, y and z block offsets assigned to the device
             * @param[in] nGPUs The number of GPUs partitioning the domain in the x, y and z directions
             **/
            __host__ [[nodiscard]] deviceIndex_t getPartitionDeviceID(
                const blockLabel_t &blockOffsets,
                const blockLabel_t &blockSpan,
                const blockLabel_t &nGPUs) noexcept
            {
                // Calculate how many blocks each GPU gets in each dimension
                // Using ceiling division to distribute blocks as evenly as possible
                const label_t blocksPerGPUx = (blockSpan.nx + nGPUs.nx) / nGPUs.nx; // ceiling division
                const label_t blocksPerGPUy = (blockSpan.ny + nGPUs.ny) / nGPUs.ny;
                const label_t blocksPerGPUz = (blockSpan.nz + nGPUs.nz) / nGPUs.nz;

                // Determine which GPU partition this block belongs to
                const label_t gpuX = blockOffsets.nx / blocksPerGPUx;
                const label_t gpuY = blockOffsets.ny / blocksPerGPUy;
                const label_t gpuZ = blockOffsets.nz / blocksPerGPUz;

                // Calculate and return GPU device ID
                return static_cast<deviceIndex_t>(gpuX + gpuY * nGPUs.nx + gpuZ * nGPUs.nx * nGPUs.ny);
            }
        };
    }
}

#endif

#endif
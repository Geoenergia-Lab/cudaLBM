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
    A class holding information about the solution grid

Namespace
    LBM::host

SourceFiles
    latticeMesh.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_HOSTLATTICEMESH_CUH
#define __MBLBM_HOSTLATTICEMESH_CUH

namespace LBM
{
    namespace host
    {
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
            /**
             * @brief Constructs a lattice mesh from program configuration
             * @param[in] programCtrl The program control object
             * @throws Error if mesh dimensions are invalid or GPU memory is insufficient
             *
             * This constructor reads mesh dimensions from the "programControl" file and performs:
             * - Validation of block decomposition compatibility
             * - Memory requirement checking for GPU
             * - Calculation of LBM relaxation parameters
             * - Initialization of device constants for GPU execution
             **/
            __host__ [[nodiscard]] latticeMesh([[maybe_unused]] const programControl &programCtrl)
                : dimensions_(string::extractParameter<host::blockLabel>("latticeMesh", "n")),
                  L_(string::extractParameter<pointVector>("latticeMesh", "L")),
                  nDevices_(string::extractParameter<host::blockLabel>("deviceDecomposition", "n"))
            {
                print();

                // Check if we are actually running GPU code
                if (programCtrl.deviceList().size() > 0)
                {
                    // Perform a block dimensions safety check
                    validate_block_dimensions(dimensions_);

                    // Safety check for the mesh dimensions
                    validate_allocation_size(programCtrl, dimensions_, nDevices_);

                    // Must be safe, so allocate device constants
                    set_constants(programCtrl, dimensions_, nBlocks(), nDevices_);
                }
            };

            /**
             * @brief Constructs a lattice mesh with specified dimensions
             * @param[in] mesh The lattice mesh
             * @param[in] meshDimensions The dimensions of the mesh to construct
             **/
            __host__ [[nodiscard]] latticeMesh(const host::latticeMesh &mesh, const host::blockLabel &meshDimensions) noexcept
                : dimensions_({meshDimensions.x, meshDimensions.y, meshDimensions.z}),
                  L_(mesh.L()),
                  nDevices_(string::extractParameter<host::blockLabel>("deviceDecomposition", "n"))
            {
                print();
            }

            /**
             * @brief Default destructor
             **/
            __host__ ~latticeMesh() noexcept {}

            /**
             * @brief Disable copying
             **/
            __host__ [[nodiscard]] latticeMesh(const latticeMesh &) = delete;
            __host__ [[nodiscard]] latticeMesh &operator=(const latticeMesh &) = delete;

            /**
             * @brief Total number of points in the mesh
             * @tparam T The size type
             **/
            __device__ __host__ [[nodiscard]] inline constexpr host::label_t size() const noexcept
            {
                return dimensions_.size();
            }

            /**
             * @brief Number of points in the mesh in a specific direction
             * @tparam alpha The axis direction (X, Y or Z)
             * @tparam T The size type
             **/
            template <axis::type alpha>
            __device__ __host__ [[nodiscard]] inline constexpr host::label_t dimension() const noexcept
            {
                return dimensions_.value<alpha>();
            }

            /**
             * @brief Number of points in the mesh in each specific direction
             **/
            __device__ __host__ [[nodiscard]] inline constexpr const host::blockLabel &dimensions() const noexcept
            {
                return dimensions_;
            }

            /**
             * @brief Number of blocks in the mesh in a specific direction
             * @tparam alpha The axis direction (X, Y or Z)
             * @tparam ValueType The size type
             * @return The number of blocks in the specified direction
             **/
            template <const axis::type alpha>
            __device__ __host__ [[nodiscard]] inline constexpr host::label_t nBlocks() const noexcept
            {
                return dimensions_.value<alpha>() / block::n<alpha, host::label_t>();
            }

            /**
             * @brief Number of blocks in the mesh in each specific direction
             * @return host::blockLabel containing the number of blocks in each direction
             **/
            __device__ __host__ [[nodiscard]] inline constexpr host::blockLabel nBlocks() const noexcept
            {
                return host::blockLabel(nBlocks<axis::X>(), nBlocks<axis::Y>(), nBlocks<axis::Z>());
            }

            /**
             * @brief Get grid dimensions for CUDA kernel launches
             * @return dim3 structure with grid dimensions
             **/
            __host__ [[nodiscard]] inline constexpr dim3 gridBlock() const noexcept
            {
                return {static_cast<uint32_t>(blocksPerDevice<axis::X>()), static_cast<uint32_t>(blocksPerDevice<axis::Y>()), static_cast<uint32_t>(blocksPerDevice<axis::Z>())};
            }

            /**
             * @brief Get thread block dimensions for CUDA kernel launches
             * @return dim3 structure with thread block dimensions
             **/
            __device__ __host__ [[nodiscard]] static inline consteval dim3 threadBlock() noexcept
            {
                return {block::nx<uint32_t>(), block::ny<uint32_t>(), block::nz<uint32_t>()};
            }

            /**
             * @brief Get physical domain dimensions
             * @return Const reference to pointVector containing domain size
             **/
            __host__ [[nodiscard]] inline constexpr const pointVector &L() const noexcept
            {
                return L_;
            }

            /**
             * @brief Boundary check for the faces
             * @param[in] x,y,z The coordinate of the point
             * @return True if the point is on the boundary, false otherwise
             **/
            __host__ [[nodiscard]] inline constexpr bool West(const host::label_t x) const noexcept
            {
                return (x == 0);
            }
            __host__ [[nodiscard]] inline constexpr bool East(const host::label_t x) const noexcept
            {
                return (x == dimensions_.x - 1);
            }
            __host__ [[nodiscard]] inline constexpr bool South(const host::label_t y) const noexcept
            {
                return (y == 0);
            }
            __host__ [[nodiscard]] inline constexpr bool North(const host::label_t y) const noexcept
            {
                return (y == dimensions_.y - 1);
            }
            __host__ [[nodiscard]] inline constexpr bool Back(const host::label_t z) const noexcept
            {
                return (z == 0);
            }
            __host__ [[nodiscard]] inline constexpr bool Front(const host::label_t z) const noexcept
            {
                return (z == dimensions_.z - 1);
            }

            /**
             * @brief Returns the number of devices
             * @tparam alpha The axis direction (X, Y or Z)
             * @tparam T The return type
             **/
            __host__ [[nodiscard]] inline constexpr const host::blockLabel &nDevices() const noexcept
            {
                return nDevices_;
            }
            template <const axis::type alpha>
            __host__ [[nodiscard]] inline constexpr host::label_t nDevices() const noexcept
            {
                return nDevices_.value<alpha>();
            }

            template <const axis::type alpha, const host::label_t QF>
            __host__ [[nodiscard]] inline constexpr host::label_t nFaces() const noexcept
            {
                axis::assertions::validate<alpha, axis::NOT_NULL>();

                return (dimensions_.size() * QF) / (block::n<alpha, host::label_t>());
            }

            /**
             * @brief Computes the allocation size along a block face for a given QF
             * @tparam alpha The axis direction (X, Y or Z)
             * @tparam T The return type
             **/
            template <const axis::type alpha, const host::label_t QF>
            __host__ [[nodiscard]] inline constexpr host::label_t nFacesPerDevice() const noexcept
            {
                axis::assertions::validate<alpha, axis::NOT_NULL>();

                return (dimensions_.size() * QF) / (block::n<alpha, host::label_t>() * nDevices<alpha>());
            }

            /**
             * @brief Computes the allocation size for the number of points per GPU
             **/
            __host__ [[nodiscard]] inline constexpr host::label_t sizePerDevice() const noexcept
            {
                const host::label_t nxPointsPerDevice = dimensions_.value<axis::X>() / nDevices<axis::X>();
                const host::label_t nyPointsPerDevice = dimensions_.value<axis::Y>() / nDevices<axis::Y>();
                const host::label_t nzPointsPerDevice = dimensions_.value<axis::Z>() / nDevices<axis::Z>();

                return nxPointsPerDevice * nyPointsPerDevice * nzPointsPerDevice;
            }

            /**
             * @brief Computes the allocation size for the number of blocks per GPU
             **/
            template <const axis::type alpha>
            __host__ [[nodiscard]] inline constexpr host::label_t blocksPerDevice() const noexcept
            {
                return nBlocks<alpha>() / nDevices_.value<alpha>();
            }
            __host__ [[nodiscard]] inline constexpr host::blockLabel blocksPerDevice() const noexcept
            {
                return {nBlocks<axis::X>() / nDevices_.value<axis::X>(), nBlocks<axis::Y>() / nDevices_.value<axis::Y>(), nBlocks<axis::Z>() / nDevices_.value<axis::Z>()};
            }

        private:
            /**
             * @brief The number of lattices in the x, y and z directions
             **/
            const host::blockLabel dimensions_;

            /**
             * @brief Physical dimensions of the domain
             **/
            const pointVector L_;

            /**
             * @brief Number of devices in the x, y and z directions
             **/
            const host::blockLabel nDevices_;

            /**
             * @brief Validates that the block decomposition is compatible with the mesh dimensions
             *
             * @param[in] nBlocks The number of blocks in each direction
             * @param[in] dimensions The dimensions of the mesh
             **/
            __host__ static void validate_block_dimensions(const host::blockLabel &dimensions)
            {
                const host::label_t nxBlocks = dimensions.x / block::nx<host::label_t>();
                const host::label_t nyBlocks = dimensions.y / block::ny<host::label_t>();
                const host::label_t nzBlocks = dimensions.z / block::nz<host::label_t>();

                if (!(block::nx<host::label_t>() * nxBlocks == dimensions.x))
                {
                    throw std::runtime_error("block::nx() * mesh.nxBlocks() not equal to mesh.dimension<axis::X>(()\nMesh dimensions should be multiples of 8");
                }
                if (!(block::ny<host::label_t>() * nyBlocks == dimensions.y))
                {
                    throw std::runtime_error("block::ny() * mesh.nyBlocks() not equal to mesh.dimension<axis::Y>()\nMesh dimensions should be multiples of 8");
                }
                if (!(block::nz<host::label_t>() * nzBlocks == dimensions.z))
                {
                    throw std::runtime_error("block::nz() * mesh.nzBlocks() not equal to mesh.dimension<axis::Z>()\nMesh dimensions should be multiples of 8");
                }
                if (!(block::nx<host::label_t>() * nxBlocks * block::ny<host::label_t>() * nyBlocks * block::nz<host::label_t>() * nzBlocks == dimensions.x * dimensions.y * dimensions.z))
                {
                    throw std::runtime_error("block::nx() * nxBlocks() * block::ny() * nyBlocks() * block::nz() * nzBlocks() not equal to mesh.size()\nMesh dimensions should be multiples of 8");
                }
            }

            /**
             * @brief Validates that the mesh dimensions do not exceed the limits of host::label_t
             * and that the per-GPU allocation size does not exceed available GPU memory
             *
             * @param[in] programCtrl The program control object containing device information
             * @param[in] dimensions The dimensions of the mesh
             * @param[in] nDevices The number of devices in each direction for multi-GPU decomposition
             **/
            static void validate_allocation_size(
                const programControl &programCtrl,
                const host::blockLabel &dimensions,
                const host::blockLabel &nDevices)
            {
                const host::label_t nxTemp = static_cast<host::label_t>(dimensions.value<axis::X>());
                const host::label_t nyTemp = static_cast<host::label_t>(dimensions.value<axis::Y>());
                const host::label_t nzTemp = static_cast<host::label_t>(dimensions.value<axis::Z>());
                const host::label_t nPointsTemp = nxTemp * nyTemp * nzTemp;
                constexpr const host::label_t typeLimit = static_cast<host::label_t>(std::numeric_limits<device::label_t>::max());

                // Check that the mesh dimensions won't overflow the type limit for host::label_t
                if (nPointsTemp >= typeLimit)
                {
                    throw std::runtime_error(
                        "\nMesh size exceeds maximum allowed value:\n"
                        "Number of mesh points: " +
                        std::to_string(nPointsTemp) +
                        "\nLimit of device::label_t: " +
                        std::to_string(typeLimit));
                }

                // Check that the mesh dimensions are not too large for GPU memory
                for (host::label_t virtualDeviceIndex = 0; virtualDeviceIndex < programCtrl.deviceList().size(); virtualDeviceIndex++)
                {
                    // Calculate the per-GPU allocation size
                    const host::label_t nxPointsPerDevice = dimensions.value<axis::X>() / nDevices.value<axis::X>();
                    const host::label_t nyPointsPerDevice = dimensions.value<axis::Y>() / nDevices.value<axis::Y>();
                    const host::label_t nzPointsPerDevice = dimensions.value<axis::Z>() / nDevices.value<axis::Z>();
                    const host::label_t nPointsPerDevice = nxPointsPerDevice * nyPointsPerDevice * nzPointsPerDevice;

                    const cudaDeviceProp props = GPU::properties(programCtrl.deviceList()[virtualDeviceIndex]);
                    const host::label_t totalMemTemp = props.totalGlobalMem;
                    const host::label_t allocationSize = nPointsPerDevice * static_cast<host::label_t>(sizeof(scalar_t)) * (NUMBER_MOMENTS<host::label_t>());

                    if (allocationSize >= totalMemTemp)
                    {
                        const double gbAllocation = static_cast<double>(allocationSize / (1024 * 1024 * 1024));
                        const double gbAvailable = static_cast<double>(totalMemTemp / (1024 * 1024 * 1024));

                        const name_t errorString = name_t("Insufficient GPU memory (") + std::to_string(gbAllocation) + name_t(" GiB requested, ") + std::to_string(gbAvailable) + name_t(" GiB available)");

                        errorHandler::check(-1, errorString);
                    }
                }
            }

            /**
             * @brief Initializes device constants for each GPU based on the program control and mesh dimensions
             * @param[in] programCtrl The program control object containing simulation parameters
             * @param[in] dimensions The dimensions of the mesh
             * @param[in] nDevices The number of devices in each direction for multi-GPU decomposition
             **/
            __host__ static void set_constants(const programControl &programCtrl, const host::blockLabel &dimensions, const host::blockLabel &nBlocks, const host::blockLabel &nDevices)
            {
                GPU::forAll(
                    nDevices,
                    [&](const host::label_t dx, const host::label_t dy, const host::label_t dz)
                    {
                        const host::label_t virtualDeviceIndex = GPU::idx(dx, dy, dz, nDevices.value<axis::X>(), nDevices.value<axis::Y>());

                        errorHandler::check(cudaSetDevice(programCtrl.deviceList()[virtualDeviceIndex]));

                        const device::label_t nx = static_cast<device::label_t>(dimensions.x);
                        const device::label_t ny = static_cast<device::label_t>(dimensions.y);
                        const device::label_t nz = static_cast<device::label_t>(dimensions.z);

                        const device::label_t nxBlocksPerDevice = static_cast<device::label_t>(nBlocks.value<axis::X>() / nDevices.value<axis::X>());
                        const device::label_t nyBlocksPerDevice = static_cast<device::label_t>(nBlocks.value<axis::Y>() / nDevices.value<axis::Y>());
                        const device::label_t nzBlocksPerDevice = static_cast<device::label_t>(nBlocks.value<axis::Z>() / nDevices.value<axis::Z>());

                        const device::label_t xBlockOffset = static_cast<device::label_t>((nBlocks.value<axis::X>() / nDevices.value<axis::X>()) * dx);
                        const device::label_t yBlockOffset = static_cast<device::label_t>((nBlocks.value<axis::X>() / nDevices.value<axis::X>()) * dy);
                        const device::label_t zBlockOffset = static_cast<device::label_t>((nBlocks.value<axis::X>() / nDevices.value<axis::X>()) * dz);

                        // Allocate mesh symbols on the GPU
                        device::copyToSymbol(device::nx, nx);
                        device::copyToSymbol(device::ny, ny);
                        device::copyToSymbol(device::nz, nz);
                        device::copyToSymbol(device::NUM_BLOCK_X, nxBlocksPerDevice);
                        device::copyToSymbol(device::NUM_BLOCK_Y, nyBlocksPerDevice);
                        device::copyToSymbol(device::NUM_BLOCK_Z, nzBlocksPerDevice);
                        device::copyToSymbol(device::BLOCK_OFFSET_X, xBlockOffset);
                        device::copyToSymbol(device::BLOCK_OFFSET_Y, yBlockOffset);
                        device::copyToSymbol(device::BLOCK_OFFSET_Z, zBlockOffset);
                    });
            }

            /**
             * @brief Prints the lattice mesh properties to the console
             **/
            __host__ inline void print() const noexcept
            {
                dimensions_.print("latticeMesh");
                std::cout << std::endl;

                L_.print("meshSize");
                std::cout << std::endl;

                host::blockLabel{block::nx(), block::ny(), block::nz()}.print("blockDimensions");
                std::cout << std::endl;

                nDevices_.print("deviceDecomposition");
                std::cout << std::endl;
            }
        };
    }
}

#endif
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
                : dimensions_(string::extractParameter<blockLabel>("latticeMesh", "n")),
                  L_(string::extractParameter<pointVector>("latticeMesh", "L")),
                  nDevices_(string::extractParameter<blockLabel>("deviceDecomposition", "n"))
            {
                print();

                // Perform a block dimensions safety check
                validate_block_dimensions(dimensions_);

                // Safety check for the mesh dimensions
                validate_allocation_size(programCtrl, dimensions_, nDevices_);

                // Must be safe, so allocate device constants
                set_constants(programCtrl, dimensions_, nBlocks(), nDevices_);
            };

            /**
             * @brief Constructs a lattice mesh with specified dimensions
             * @param[in] mesh The lattice mesh
             * @param[in] meshDimensions The dimensions of the mesh to construct
             **/
            __host__ [[nodiscard]] latticeMesh(const host::latticeMesh &mesh, const blockLabel &meshDimensions) noexcept
                : dimensions_({meshDimensions.x, meshDimensions.y, meshDimensions.z}),
                  L_(mesh.L()),
                  nDevices_(string::extractParameter<blockLabel>("deviceDecomposition", "n"))
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
            template <typename ValueType = device::label_t>
            __device__ __host__ [[nodiscard]] inline constexpr ValueType size() const noexcept
            {
                return dimensions_.size<ValueType>();
            }

            /**
             * @brief Number of points in the mesh in a specific direction
             * @tparam alpha The axis type (X, Y or Z)
             * @tparam T The size type
             **/
            template <axis::type alpha, typename ValueType = device::label_t>
            __device__ __host__ [[nodiscard]] inline constexpr ValueType dimension() const noexcept
            {
                return dimensions_.value<alpha, ValueType>();
            }

            /**
             * @brief Number of points in the mesh in each specific direction
             **/
            __device__ __host__ [[nodiscard]] inline constexpr const blockLabel &dimensions() const noexcept
            {
                return dimensions_;
            }

            /**
             * @brief Number of blocks in the mesh in a specific direction
             * @tparam alpha The axis type (X, Y or Z)
             * @tparam ValueType The size type
             * @return The number of blocks in the specified direction
             **/
            template <const axis::type alpha, typename ValueType = device::label_t>
            __device__ __host__ [[nodiscard]] inline constexpr ValueType nBlocks() const noexcept
            {
                return dimensions_.value<alpha, ValueType>() / block::n<alpha, ValueType>();
            }

            /**
             * @brief Number of blocks in the mesh in each specific direction
             * @return blockLabel containing the number of blocks in each direction
             **/
            __device__ __host__ [[nodiscard]] inline constexpr blockLabel nBlocks() const noexcept
            {
                return blockLabel(nBlocks<axis::X, blockLabel::value_type>(), nBlocks<axis::Y, blockLabel::value_type>(), nBlocks<axis::Z, blockLabel::value_type>());
            }

            /**
             * @brief Get grid dimensions for CUDA kernel launches
             * @return dim3 structure with grid dimensions
             **/
            __host__ [[nodiscard]] inline constexpr dim3 gridBlock() const noexcept
            {
                return {blocksPerDevice<axis::X, uint32_t>(), blocksPerDevice<axis::Y, uint32_t>(), blocksPerDevice<axis::Z, uint32_t>()};
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
            __host__ [[nodiscard]] inline constexpr bool West(const device::label_t x) const noexcept
            {
                return (x == 0);
            }
            __host__ [[nodiscard]] inline constexpr bool East(const device::label_t x) const noexcept
            {
                return (x == dimensions_.x - 1);
            }
            __host__ [[nodiscard]] inline constexpr bool South(const device::label_t y) const noexcept
            {
                return (y == 0);
            }
            __host__ [[nodiscard]] inline constexpr bool North(const device::label_t y) const noexcept
            {
                return (y == dimensions_.y - 1);
            }
            __host__ [[nodiscard]] inline constexpr bool Back(const device::label_t z) const noexcept
            {
                return (z == 0);
            }
            __host__ [[nodiscard]] inline constexpr bool Front(const device::label_t z) const noexcept
            {
                return (z == dimensions_.z - 1);
            }

            /**
             * @brief Returns the number of devices
             * @tparam alpha The axis (X, Y or Z)
             * @tparam T The return type
             **/
            __host__ [[nodiscard]] inline constexpr const blockLabel &nDevices() const noexcept
            {
                return nDevices_;
            }
            template <const axis::type alpha, typename ValueType = device::label_t>
            __host__ [[nodiscard]] inline constexpr ValueType nDevices() const noexcept
            {
                return nDevices_.value<alpha, ValueType>();
            }

            template <const axis::type alpha, const device::label_t QF, typename ValueType = device::label_t>
            __host__ [[nodiscard]] inline constexpr ValueType nFaces() const noexcept
            {
                axis::assertions::validate<alpha, axis::NOT_NULL>();

                return (dimensions_.size<ValueType>() * static_cast<ValueType>(QF)) / (block::n<alpha, ValueType>());
            }

            /**
             * @brief Computes the allocation size along a block face for a given QF
             * @tparam alpha The axis (X, Y or Z)
             * @tparam T The return type
             **/
            template <const axis::type alpha, const device::label_t QF, typename ValueType = device::label_t>
            __host__ [[nodiscard]] inline constexpr ValueType nFacesPerDevice() const noexcept
            {
                axis::assertions::validate<alpha, axis::NOT_NULL>();

                return (dimensions_.size<ValueType>() * static_cast<ValueType>(QF)) / (block::n<alpha, ValueType>() * nDevices<alpha, ValueType>());
            }

            /**
             * @brief Computes the allocation size for the number of points per GPU
             **/
            template <typename ValueType = device::label_t>
            __host__ [[nodiscard]] inline constexpr ValueType sizePerDevice() const noexcept
            {
                const ValueType nxPointsPerDevice = dimensions_.value<axis::X, ValueType>() / nDevices<axis::X, ValueType>();
                const ValueType nyPointsPerDevice = dimensions_.value<axis::Y, ValueType>() / nDevices<axis::Y, ValueType>();
                const ValueType nzPointsPerDevice = dimensions_.value<axis::Z, ValueType>() / nDevices<axis::Z, ValueType>();

                return nxPointsPerDevice * nyPointsPerDevice * nzPointsPerDevice;
            }

            /**
             * @brief Computes the allocation size for the number of blocks per GPU
             **/
            template <const axis::type alpha, typename ValueType = device::label_t>
            __host__ [[nodiscard]] inline constexpr ValueType blocksPerDevice() const noexcept
            {
                return nBlocks<alpha, ValueType>() / nDevices_.value<alpha, ValueType>();
            }
            __host__ [[nodiscard]] inline constexpr blockLabel blocksPerDevice() const noexcept
            {
                return {nBlocks<axis::X>() / nDevices_.value<axis::X>(), nBlocks<axis::Y>() / nDevices_.value<axis::Y>(), nBlocks<axis::Z>() / nDevices_.value<axis::Z>()};
            }

        private:
            /**
             * @brief The number of lattices in the x, y and z directions
             **/
            const blockLabel dimensions_;

            /**
             * @brief Physical dimensions of the domain
             **/
            const pointVector L_;

            /**
             * @brief Number of devices in the x, y and z directions
             **/
            const blockLabel nDevices_;

            /**
             * @brief Validates that the block decomposition is compatible with the mesh dimensions
             *
             * @param[in] nBlocks The number of blocks in each direction
             * @param[in] dimensions The dimensions of the mesh
             **/
            __host__ static void validate_block_dimensions(const blockLabel &dimensions)
            {
                const device::label_t nxBlocks = dimensions.x / block::nx();
                const device::label_t nyBlocks = dimensions.y / block::ny();
                const device::label_t nzBlocks = dimensions.z / block::nz();

                if (!(block::nx() * nxBlocks == dimensions.x))
                {
                    throw std::runtime_error("block::nx() * mesh.nxBlocks() not equal to mesh.dimension<axis::X>(()\nMesh dimensions should be multiples of 8");
                }
                if (!(block::ny() * nyBlocks == dimensions.y))
                {
                    throw std::runtime_error("block::ny() * mesh.nyBlocks() not equal to mesh.dimension<axis::Y>()\nMesh dimensions should be multiples of 8");
                }
                if (!(block::nz() * nzBlocks == dimensions.z))
                {
                    throw std::runtime_error("block::nz() * mesh.nzBlocks() not equal to mesh.dimension<axis::Z>()\nMesh dimensions should be multiples of 8");
                }
                if (!(block::nx() * nxBlocks * block::ny() * nyBlocks * block::nz() * nzBlocks == dimensions.x * dimensions.y * dimensions.z))
                {
                    throw std::runtime_error("block::nx() * nxBlocks() * block::ny() * nyBlocks() * block::nz() * nzBlocks() not equal to mesh.size()\nMesh dimensions should be multiples of 8");
                }
            }

            /**
             * @brief Validates that the mesh dimensions do not exceed the limits of device::label_t
             * and that the per-GPU allocation size does not exceed available GPU memory
             *
             * @param[in] programCtrl The program control object containing device information
             * @param[in] dimensions The dimensions of the mesh
             * @param[in] nDevices The number of devices in each direction for multi-GPU decomposition
             **/
            static void validate_allocation_size(
                const programControl &programCtrl,
                const blockLabel &dimensions,
                const blockLabel &nDevices)
            {
                const host::label_t nxTemp = static_cast<host::label_t>(dimensions.value<axis::X>());
                const host::label_t nyTemp = static_cast<host::label_t>(dimensions.value<axis::Y>());
                const host::label_t nzTemp = static_cast<host::label_t>(dimensions.value<axis::Z>());
                const host::label_t nPointsTemp = nxTemp * nyTemp * nzTemp;
                constexpr const host::label_t typeLimit = static_cast<host::label_t>(std::numeric_limits<device::label_t>::max());

                // Check that the mesh dimensions won't overflow the type limit for device::label_t
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
                    const device::label_t nxPointsPerDevice = dimensions.value<axis::X>() / nDevices.value<axis::X>();
                    const device::label_t nyPointsPerDevice = dimensions.value<axis::Y>() / nDevices.value<axis::Y>();
                    const device::label_t nzPointsPerDevice = dimensions.value<axis::Z>() / nDevices.value<axis::Z>();
                    const device::label_t nPointsPerDevice = nxPointsPerDevice * nyPointsPerDevice * nzPointsPerDevice;

                    const cudaDeviceProp props = GPU::properties(programCtrl.deviceList()[virtualDeviceIndex]);
                    const host::label_t totalMemTemp = static_cast<host::label_t>(props.totalGlobalMem);
                    const host::label_t allocationSize = nPointsPerDevice * static_cast<host::label_t>(sizeof(scalar_t)) * (NUMBER_MOMENTS<host::label_t>());

                    if (allocationSize >= totalMemTemp)
                    {
                        const double gbAllocation = static_cast<double>(allocationSize / (1024 * 1024 * 1024));
                        const double gbAvailable = static_cast<double>(totalMemTemp / (1024 * 1024 * 1024));

                        throw std::runtime_error(
                            "\nInsufficient GPU memory:\nAttempted to allocate: " +
                            std::to_string(allocationSize) +
                            " bytes (" +
                            std::to_string(gbAllocation) +
                            " GB)\n"
                            "Available GPU memory: " +
                            std::to_string(totalMemTemp) +
                            " bytes (" +
                            std::to_string(gbAvailable) +
                            " GB)");
                    }
                }
            }

            /**
             * @brief Initializes device constants for each GPU based on the program control and mesh dimensions
             * @param[in] programCtrl The program control object containing simulation parameters
             * @param[in] dimensions The dimensions of the mesh
             * @param[in] nDevices The number of devices in each direction for multi-GPU decomposition
             **/
            __host__ static void set_constants(const programControl &programCtrl, const blockLabel &dimensions, const blockLabel &nBlocks, const blockLabel &nDevices)
            {
                GPU::forAll(
                    nDevices,
                    [&](const device::label_t dx, const device::label_t dy, const device::label_t dz)
                    {
                        const device::label_t virtualDeviceIndex = GPU::idx(dx, dy, dz, nDevices.value<axis::X>(), nDevices.value<axis::Y>());

                        errorHandler::check(cudaSetDevice(programCtrl.deviceList()[virtualDeviceIndex]));

                        const device::label_t nxBlocksPerDevice = nBlocks.value<axis::X>() / nDevices.value<axis::X>();
                        const device::label_t nyBlocksPerDevice = nBlocks.value<axis::Y>() / nDevices.value<axis::Y>();
                        const device::label_t nzBlocksPerDevice = nBlocks.value<axis::Z>() / nDevices.value<axis::Z>();

                        // Allocate mesh symbols on the GPU
                        device::copyToSymbol(device::nx, dimensions.x);
                        device::copyToSymbol(device::ny, dimensions.y);
                        device::copyToSymbol(device::nz, dimensions.z);
                        device::copyToSymbol(device::NUM_BLOCK_X, nxBlocksPerDevice);
                        device::copyToSymbol(device::NUM_BLOCK_Y, nyBlocksPerDevice);
                        device::copyToSymbol(device::NUM_BLOCK_Z, nzBlocksPerDevice);
                        device::copyToSymbol(device::BLOCK_OFFSET_X, nxBlocksPerDevice * dx);
                        device::copyToSymbol(device::BLOCK_OFFSET_Y, nyBlocksPerDevice * dy);
                        device::copyToSymbol(device::BLOCK_OFFSET_Z, nzBlocksPerDevice * dz);
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

                blockLabel{block::nx(), block::ny(), block::nz()}.print("blockDimensions");
                std::cout << std::endl;

                nDevices_.print("deviceDecomposition");
                std::cout << std::endl;
            }
        };
    }
}

#endif
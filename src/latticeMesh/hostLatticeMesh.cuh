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
            __host__ [[nodiscard]] latticeMesh(const programControl &programCtrl)
                : dimensions_(string::extractParameter<host::blockLabel>("latticeMesh", "n")),
                  L_(string::extractParameter<pointVector>("latticeMesh", "L")),
                  nDevices_(string::extractParameter<host::blockLabel>("deviceDecomposition", "n")),
                  gridBlock_(initialiseGridBlock())
            {
                print();

                // Check if we are actually running GPU code
                if (programCtrl.deviceList().size() > 0)
                {
                    // Perform a block dimensions safety check
                    validate_block_dimensions();

                    // Safety check for the mesh dimensions
                    validate_allocation_size(dimensions_);

                    // Must be safe, so allocate device constants
                    set_constants(programCtrl);
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
                  nDevices_(string::extractParameter<host::blockLabel>("deviceDecomposition", "n")),
                  gridBlock_(initialiseGridBlock()) {}

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
             * @brief Returns the total number of lattice points in the mesh.
             *
             * @return The total number of grid points.
             **/
            __host__ [[nodiscard]] inline constexpr host::label_t size() const noexcept
            {
                return dimensions_.size();
            }

            /**
             * @brief Returns the number of lattice points along a given axis.
             *
             * @tparam alpha Axis direction to query.
             * @return Number of points in the selected direction.
             **/
            template <const axis::type alpha>
            __host__ [[nodiscard]] inline constexpr host::label_t dimension() const noexcept
            {
                return dimensions_.value<alpha>();
            }

            /**
             * @brief Returns the mesh dimensions in all three directions.
             *
             * @return Const reference to the mesh extents.
             **/
            __host__ [[nodiscard]] inline constexpr const host::blockLabel &dimensions() const noexcept
            {
                return dimensions_;
            }

            /**
             * @brief Returns the number of blocks along a given axis.
             *
             * @tparam alpha Axis direction to query.
             * @return Number of blocks in the selected direction.
             **/
            template <const axis::type alpha>
            __host__ [[nodiscard]] inline constexpr host::label_t nBlocks() const noexcept
            {
                return dimensions_.value<alpha>() / block::n<alpha, host::label_t>();
            }

            /**
             * @brief Returns the number of blocks in each direction.
             *
             * @return Block counts for x, y, and z directions.
             **/
            __host__ [[nodiscard]] inline constexpr host::blockLabel nBlocks() const noexcept
            {
                return host::blockLabel(nBlocks<axis::X>(), nBlocks<axis::Y>(), nBlocks<axis::Z>());
            }

            /**
             * @brief Returns the CUDA grid dimensions used for kernel launches.
             *
             * @return Grid dimensions for the three launch phases.
             **/
            __host__ [[nodiscard]] inline constexpr const std::array<dim3, 3> &gridBlock() const noexcept
            {
                return gridBlock_;
            }

            /**
             * @brief Returns the default CUDA thread-block dimensions for kernel launches.
             *
             * @return Thread block shape.
             **/
            __host__ [[nodiscard]] static inline consteval dim3 threadBlock() noexcept
            {
                return {block::nx<uint32_t>(), block::ny<uint32_t>(), block::nz<uint32_t>()};
            }

            /**
             * @brief Returns the physical domain size associated with the mesh.
             *
             * @return Const reference to the physical length vector.
             **/
            __host__ [[nodiscard]] inline constexpr const pointVector &L() const noexcept
            {
                return L_;
            }

            /**
             * @brief Tests whether the coordinate lies on the west boundary.
             *
             * @param[in] x X coordinate of the point.
             * @return true if the point is on the west face; otherwise false.
             **/
            __host__ [[nodiscard]] inline constexpr bool West(const host::label_t x) const noexcept
            {
                return (x == 0);
            }

            /**
             * @brief Tests whether the coordinate lies on the east boundary.
             *
             * @param[in] x X coordinate of the point.
             * @return true if the point is on the east face; otherwise false.
             **/
            __host__ [[nodiscard]] inline constexpr bool East(const host::label_t x) const noexcept
            {
                return (x == dimensions_.x - 1);
            }

            /**
             * @brief Tests whether the coordinate lies on the south boundary.
             *
             * @param[in] y Y coordinate of the point.
             * @return true if the point is on the south face; otherwise false.
             **/
            __host__ [[nodiscard]] inline constexpr bool South(const host::label_t y) const noexcept
            {
                return (y == 0);
            }

            /**
             * @brief Tests whether the coordinate lies on the north boundary.
             *
             * @param[in] y Y coordinate of the point.
             * @return true if the point is on the north face; otherwise false.
             **/
            __host__ [[nodiscard]] inline constexpr bool North(const host::label_t y) const noexcept
            {
                return (y == dimensions_.y - 1);
            }

            /**
             * @brief Tests whether the coordinate lies on the back boundary.
             *
             * @param[in] z Z coordinate of the point.
             * @return true if the point is on the back face; otherwise false.
             **/
            __host__ [[nodiscard]] inline constexpr bool Back(const host::label_t z) const noexcept
            {
                return (z == 0);
            }

            /**
             * @brief Tests whether the coordinate lies on the front boundary.
             *
             * @param[in] z Z coordinate of the point.
             * @return true if the point is on the front face; otherwise false.
             **/
            __host__ [[nodiscard]] inline constexpr bool Front(const host::label_t z) const noexcept
            {
                return (z == dimensions_.z - 1);
            }

            /**
             * @brief Returns the per-axis device decomposition counts.
             *
             * @return Device counts in each direction.
             **/
            __host__ [[nodiscard]] inline constexpr const host::blockLabel &nDevices() const noexcept
            {
                return nDevices_;
            }

            /**
             * @brief Returns the device count along a given axis.
             *
             * @tparam alpha Axis direction to query.
             * @return Number of devices assigned along the selected axis.
             **/
            template <const axis::type alpha>
            __host__ [[nodiscard]] inline constexpr host::label_t nDevices() const noexcept
            {
                return nDevices_.value<alpha>();
            }

            /**
             * @brief Returns the halo size on a face for a given number of velocities.
             *
             * @tparam alpha Axis direction to inspect.
             * @tparam QF Number of face values stored along the face.
             * @return Number of entries needed for the face halo.
             **/
            template <const axis::type alpha, const host::label_t QF>
            __host__ [[nodiscard]] inline constexpr host::label_t nFaces() const noexcept
            {
                axis::assertions::validate<alpha, axis::NOT_NULL>();

                return (dimensions_.size() * QF) / (block::n<alpha, host::label_t>());
            }

            /**
             * @brief Returns the per-device halo size for a given axis and face width.
             *
             * @tparam alpha Axis direction to inspect.
             * @tparam QF Number of face values stored along the face.
             * @return Number of face entries assigned to each device.
             **/
            template <const axis::type alpha, const host::label_t QF>
            __host__ [[nodiscard]] inline constexpr host::label_t nFacesPerDevice() const noexcept
            {
                axis::assertions::validate<alpha, axis::NOT_NULL>();

                return (dimensions_.size() * QF) / (block::n<alpha, host::label_t>() * nDevices<alpha>());
            }

            /**
             * @brief Returns the number of lattice points assigned to each device.
             *
             * @return Number of mesh points per GPU.
             **/
            __host__ [[nodiscard]] inline constexpr host::label_t sizePerDevice() const noexcept
            {
                const host::label_t nxPointsPerDevice = dimensions_.value<axis::X>() / nDevices<axis::X>();
                const host::label_t nyPointsPerDevice = dimensions_.value<axis::Y>() / nDevices<axis::Y>();
                const host::label_t nzPointsPerDevice = dimensions_.value<axis::Z>() / nDevices<axis::Z>();

                return nxPointsPerDevice * nyPointsPerDevice * nzPointsPerDevice;
            }

            /**
             * @brief Returns the number of blocks assigned to each device along a specific axis.
             *
             * @tparam alpha Axis direction to inspect.
             * @return Number of blocks per GPU in the selected direction.
             **/
            template <const axis::type alpha>
            __host__ [[nodiscard]] inline constexpr host::label_t blocksPerDevice() const noexcept
            {
                return nBlocks<alpha>() / nDevices_.value<alpha>();
            }

            /**
             * @brief Returns the number of blocks per device in each direction.
             *
             * @return Block counts per GPU for x, y, and z directions.
             **/
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
             * @brief Grid dimensions for CUDA kernel launches, calculated based on block decomposition and device count
             **/
            const std::array<dim3, 3> gridBlock_;

            /**
             * @brief Builds the CUDA grid launch dimensions from the per-device block layout.
             *
             * @return Three grid descriptors used by the solver kernels.
             **/
            __host__ [[nodiscard]] inline constexpr const std::array<dim3, 3> initialiseGridBlock() const noexcept
            {
                return {
                    dim3(static_cast<uint32_t>(blocksPerDevice<axis::X>()), static_cast<uint32_t>(blocksPerDevice<axis::Y>()), static_cast<uint32_t>(1)),
                    dim3(static_cast<uint32_t>(blocksPerDevice<axis::X>()), static_cast<uint32_t>(blocksPerDevice<axis::Y>()), static_cast<uint32_t>(blocksPerDevice<axis::Z>() - 2)),
                    dim3(static_cast<uint32_t>(blocksPerDevice<axis::X>()), static_cast<uint32_t>(blocksPerDevice<axis::Y>()), static_cast<uint32_t>(1))};
            }

            /**
             * @brief Validates that the mesh dimensions are compatible with the CUDA block size.
             *
             * @param[in] dimensions Mesh extents to validate.
             *
             * @throws std::runtime_error If the mesh dimensions are not multiples of the block dimensions.
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
             * @brief Validates the current mesh against the block layout.
             **/
            __host__ inline void validate_block_dimensions() const
            {
                validate_block_dimensions(dimensions_);
            }

            /**
             * @brief Validates that the mesh size and per-device allocation size are safe.
             *
             * @param[in] programCtrl Program configuration used to inspect device layout.
             * @param[in] dimensions Mesh dimensions to validate.
             * @param[in] nDevices Device decomposition layout.
             *
             * @details Ensures the mesh does not exceed the storage limits of `host::label_t`
             * and checks the allocation footprint implied by the multi-GPU decomposition.
             **/
            __host__ static void validate_allocation_size(const host::blockLabel &dimensions)
            {
                const host::label_t nxTemp = static_cast<host::label_t>(dimensions.value<axis::X>());
                const host::label_t nyTemp = static_cast<host::label_t>(dimensions.value<axis::Y>());
                const host::label_t nzTemp = static_cast<host::label_t>(dimensions.value<axis::Z>());
                const host::label_t nPointsTemp = nxTemp * nyTemp * nzTemp;
                constexpr const host::label_t typeLimit = static_cast<host::label_t>(std::numeric_limits<device::label_t>::max());

                // Check that the mesh dimensions won't overflow the type limit for host::label_t
                if (nPointsTemp >= typeLimit)
                {
                    errorHandler::handle(runTime::error::LABEL_T_CAPACITY_EXCEEDED);
                }
            }

            /**
             * @brief Copies the mesh constants onto each GPU used by the simulation.
             *
             * @param[in] programCtrl Program configuration controlling the active devices.
             * @param[in] dimensions Mesh dimensions.
             * @param[in] nBlocks Number of blocks in each direction.
             * @param[in] nDevices Device decomposition across the domain.
             *
             * @details Synchronises each device and uploads the mesh metadata needed by kernels.
             **/
            __host__ static void set_constants(
                const programControl &programCtrl,
                const host::blockLabel &dimensions,
                const host::blockLabel &nBlocks,
                const host::blockLabel &nDevices)
            {
                GPU::forAll(
                    nDevices,
                    [&](const host::label_t dx, const host::label_t dy, const host::label_t dz)
                    {
                        const host::label_t virtualDeviceIndex = GPU::idx(dx, dy, dz, nDevices.value<axis::X>(), nDevices.value<axis::Y>());

                        errorHandler::handle(cudaSetDevice(programCtrl.deviceList()[virtualDeviceIndex]));

                        const device::label_t nx = static_cast<device::label_t>(dimensions.x);
                        const device::label_t ny = static_cast<device::label_t>(dimensions.y);
                        const device::label_t nz = static_cast<device::label_t>(dimensions.z);

                        const device::label_t nxBlocksPerDevice = static_cast<device::label_t>(nBlocks.value<axis::X>() / nDevices.value<axis::X>());
                        const device::label_t nyBlocksPerDevice = static_cast<device::label_t>(nBlocks.value<axis::Y>() / nDevices.value<axis::Y>());
                        const device::label_t nzBlocksPerDevice = static_cast<device::label_t>(nBlocks.value<axis::Z>() / nDevices.value<axis::Z>());

                        const device::label_t xBlockOffset = static_cast<device::label_t>((nBlocks.value<axis::X>() / nDevices.value<axis::X>()) * dx);
                        const device::label_t yBlockOffset = static_cast<device::label_t>((nBlocks.value<axis::Y>() / nDevices.value<axis::Y>()) * dy);
                        const device::label_t zBlockOffset = static_cast<device::label_t>((nBlocks.value<axis::Z>() / nDevices.value<axis::Z>()) * dz);

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
             * @brief Copies the current mesh constants onto the active GPU devices.
             *
             * @param[in] programCtrl Program configuration used to determine the active GPU layout.
             **/
            __host__ inline void set_constants(const programControl &programCtrl) const
            {
                set_constants(programCtrl, dimensions_, nBlocks(), nDevices_);
            }

            /**
             * @brief Prints the lattice mesh properties to the console
             **/
            __host__ inline void print() const noexcept
            {
                dimensions_.print<true>("latticeMesh");

                L_.print<true>("meshSize");

                host::blockLabel{block::nx<host::label_t>(), block::ny<host::label_t>(), block::nz<host::label_t>()}.print<true>("blockDimensions");

                nDevices_.print<true>("deviceDecomposition");
            }
        };
    }
}

#endif
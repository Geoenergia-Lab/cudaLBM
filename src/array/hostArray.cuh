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
    A templated class for allocating arrays on the CPU

Namespace
    LBM::host

SourceFiles
    hostArray.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_HOSTARRAY_CUH
#define __MBLBM_HOSTARRAY_CUH

namespace LBM
{
    namespace host
    {
        /**
         * @class array
         * @brief Templated RAII wrapper for host memory management with field initialization
         * @brief Allocates pinned or paged memory
         * @tparam AllocationType Allocate pinned or unpinned host memory
         * @tparam T Data type of array elements
         * @tparam VelocitySet Velocity set configuration for LBM simulation
         * @tparam TimeType Instantaneous or time-averaged field
         **/
        template <const host::mallocType AllocationType, typename T, class VelocitySet, const time::type TimeType>
        class array;

        /**
         * @class array
         * @brief Templated RAII wrapper for host memory management with field initialization
         * @brief Allocates pinned memory
         * @tparam T Data type of array elements
         * @tparam VelocitySet Velocity set configuration for LBM simulation
         * @tparam TimeType Instantaneous or time-averaged field
         **/
        template <typename T, class VelocitySet, const time::type TimeType>
        class array<host::PINNED, T, VelocitySet, TimeType>
        {
        public:
            /**
             * @brief Construct from a number of points and zero-initialise everything
             * @param nPoints The size of the memory in points to allocate
             **/
            __host__ [[nodiscard]] array(const label_t nPoints, const host::latticeMesh &mesh)
                : ptr_(host::allocate<T>(nPoints, 0)),
                  nPoints_(nPoints),
                  mesh_(mesh),
                  name_(""){};

            /**
             * @brief Construct from a number of points and uniform-initialise everything
             * @param nPoints The size of the memory in points to allocate
             * @param val The value to assign to the array
             **/
            __host__ [[nodiscard]] array(const label_t nPoints, const T val, const host::latticeMesh &mesh)
                : ptr_(host::allocate<T>(nPoints, val)),
                  nPoints_(nPoints),
                  mesh_(mesh),
                  name_(""){};

            /**
             * @brief Destructor - automatically releases device memory
             * @note Noexcept guarantee: failsafe if cudaFree fails
             **/
            ~array() noexcept
            {
                if constexpr (verbose())
                {
                    std::cout << "Freeing ptr" << std::endl;
                }
                checkCudaErrors(cudaFreeHost(ptr_));
                if constexpr (verbose())
                {
                    std::cout << "Freed ptr" << std::endl;
                }
            }

            /**
             * @brief Get read-only access to underlying data
             * @return Const pointer to device memory
             **/
            __host__ [[nodiscard]] inline constexpr T *operator()() const noexcept
            {
                return ptr_;
            }

            /**
             * @brief Unified element access (compile-time or runtime)
             * @tparam Index Type of index (integral type or std::integral_constant)
             * @param idx Index value or compile-time index tag
             * @return Reference to element at specified index
             * @pre Index must be in range [0, N-1]
             * @note Compile-time bounds checking for integral_constant types
             * @note Runtime access for integral types (no bounds checking)
             **/
            __host__ [[nodiscard]] inline constexpr T &operator[](const label_t idx) __restrict__ noexcept
            {
                // Runtime index
                return ptr_[idx];
            }

            /**
             * @brief Unified read-only element access (compile-time or runtime)
             * @tparam Index Type of index (integral type or std::integral_constant)
             * @param idx Index value or compile-time index tag
             * @return Const reference to element at specified index
             * @pre Index must be in range [0, N-1]
             * @note Compile-time bounds checking for integral_constant types
             * @note Runtime access for integral types (no bounds checking)
             **/
            __host__ [[nodiscard]] inline constexpr const T &operator[](const label_t idx) __restrict__ const noexcept
            {
                return ptr_[idx];
            }

            /**
             * @brief Get total number of elements
             * @return Number of elements in array
             **/
            __host__ [[nodiscard]] inline constexpr label_t size() const noexcept
            {
                return nPoints_;
            }

            /**
             * @brief Get read-only access to underlying data
             * @return Const pointer to device memory
             **/
            __host__ [[nodiscard]] inline constexpr const T *data() const noexcept
            {
                return ptr_;
            }

            /**
             * @brief Get mutable access to underlying data
             * @return Pointer to device memory
             **/
            __host__ [[nodiscard]] inline constexpr T *data() noexcept
            {
                return ptr_;
            }

            /**
             * @brief Copies the data from a collection of pointers on the device to the host
             * @param devPtrs The collection of pointers to device memory
             * @param mesh The lattice mesh
             **/
            template <const label_t N>
            __host__ void copy_from_device(const device::ptrCollection<N, T> &devPtrs, const host::latticeMesh &mesh)
            {
                // Should check that mesh.nPoints() * N is less than or equal to nPoints_

                if (mesh.nPoints() * N > nPoints_)
                {
                    throw std::runtime_error("Insufficient host array size");
                }

                for (label_t field = 0; field < N; field++)
                {
                    host::to_host(devPtrs[field], ptr_, field, mesh.nPoints());
                }
            }

            /**
             * @brief Get the mesh
             * @return Const reference to mesh
             **/
            __host__ [[nodiscard]] inline constexpr const host::latticeMesh &mesh() const noexcept
            {
                return mesh_;
            }

            /**
             * @brief Get field name identifier
             * @return Const reference to name string
             **/
            __host__ [[nodiscard]] inline constexpr const std::string &name() const noexcept
            {
                return name_;
            }

        private:
            /**
             * @brief Pointer to the data
             **/
            T *const ptrRestrict ptr_;

            /**
             * @brief Size of the data allocation
             **/
            const label_t nPoints_;

            /**
             * @brief Reference to the lattice mesh
             **/
            const host::latticeMesh &mesh_;

            const std::string name_;
        };

        /**
         * @class array
         * @brief Templated RAII wrapper for host memory management with field initialization
         * @brief Allocates unpinned memory
         * @tparam T Data type of array elements
         * @tparam VelocitySet Velocity set configuration for LBM simulation
         * @tparam TimeType Instantaneous or time-averaged field
         **/
        template <typename T, class VelocitySet, const time::type TimeType>
        class array<host::PAGED, T, VelocitySet, TimeType>
        {
        public:
            /**
             * @brief Constructs a host array with field initialization
             * @param[in] name Name identifier for the field
             * @param[in] mesh Lattice mesh defining array dimensions
             * @param[in] programCtrl Program control parameters
             * @post Array is initialized from latest time step or initial conditions
             **/
            __host__ [[nodiscard]] array(
                const std::string &name,
                const host::latticeMesh &mesh,
                const programControl &programCtrl)
                : arr_(initialise_array(mesh, name, programCtrl)),
                  name_(name),
                  mesh_(mesh){};

            /**
             * @brief Destructor for the host array class
             **/
            ~array() {};

            /**
             * @brief Unified element access (compile-time or runtime)
             * @tparam Index Type of index (integral type or std::integral_constant)
             * @param idx Index value or compile-time index tag
             * @return Reference to element at specified index
             * @pre Index must be in range [0, N-1]
             * @note Compile-time bounds checking for integral_constant types
             * @note Runtime access for integral types (no bounds checking)
             **/
            __host__ [[nodiscard]] inline constexpr T &operator[](const label_t idx) __restrict__ noexcept
            {
                // Runtime index
                return arr_[idx];
            }

            /**
             * @brief Unified read-only element access (compile-time or runtime)
             * @tparam Index Type of index (integral type or std::integral_constant)
             * @param idx Index value or compile-time index tag
             * @return Const reference to element at specified index
             * @pre Index must be in range [0, N-1]
             * @note Compile-time bounds checking for integral_constant types
             * @note Runtime access for integral types (no bounds checking)
             **/
            __host__ [[nodiscard]] inline constexpr const T &operator[](const label_t idx) __restrict__ const noexcept
            {
                return arr_[idx];
            }

            /**
             * @brief Get read-only access to underlying data
             * @return Const reference to data vector
             **/
            __host__ [[nodiscard]] inline constexpr const std::vector<T> &arr() const noexcept
            {
                return arr_;
            }

            /**
             * @brief Get field name identifier
             * @return Const reference to name string
             **/
            __host__ [[nodiscard]] inline constexpr const std::string &name() const noexcept
            {
                return name_;
            }

            /**
             * @brief Get the mesh
             * @return Const reference to mesh
             **/
            __host__ [[nodiscard]] inline constexpr const host::latticeMesh &mesh() const noexcept
            {
                return mesh_;
            }

            __host__ [[nodiscard]] inline consteval time::type timeType() const noexcept
            {
                return TimeType;
            }

        private:
            /**
             * @brief The underlying std::vector
             **/
            const std::vector<T> arr_;

            /**
             * @brief Names of the solution variable
             **/
            const std::string name_;

            /**
             * @brief Reference to the lattice mesh
             **/
            const host::latticeMesh mesh_;

            /**
             * @brief Initialize array from file or initial conditions
             * @param[in] mesh Lattice mesh for dimensioning
             * @param[in] fieldName Name of field to initialize
             * @param[in] programCtrl Program control parameters
             * @return Initialized data vector
             * @throws std::runtime_error if file operations fail
             **/
            __host__ [[nodiscard]] const std::vector<T> initialise_array(const host::latticeMesh &mesh, const std::string &fieldName, const programControl &programCtrl)
            {
                if (fileIO::hasIndexedFiles(programCtrl.caseName()))
                {
                    const std::string fileName = programCtrl.caseName() + "_" + std::to_string(fileIO::latestTime(programCtrl.caseName())) + ".LBMBin";

                    return fileIO::readFieldByName<T>(fileName, fieldName);
                }
                else
                {
                    return initialConditions(mesh, fieldName);
                }
            }

            /**
             * @brief Attempts to initialise the data from the case name
             * @param caseName The name of the case
             * @param mesh The lattice mesh
             * @param time The time step to initialise from
             **/
            __host__ [[nodiscard]] const std::vector<T> initialise_array(
                const std::string &caseName,
                const host::latticeMesh &mesh,
                const label_t time)
            {
                if (fileIO::hasIndexedFiles(caseName))
                {
                    // Should take the field name rather than the case name
                    const std::string fileName = caseName + "_" + std::to_string(time) + ".LBMBin";

                    return fileIO::readFieldByName<T>(fileName, caseName);
                }
                else
                {
                    // Should throw if not found
                    return initialConditions(mesh, caseName);
                }
            }

            /**
             * @brief Apply initial conditions with boundary handling
             * @param[in] mesh Lattice mesh for dimensioning and boundary detection
             * @param[in] fieldName Name of field for boundary condition lookup
             * @return Initialized data vector with boundary conditions applied
             **/
            __host__ [[nodiscard]] const std::vector<T> initialConditions(const host::latticeMesh &mesh, const std::string &fieldName)
            {
                const boundaryFields<VelocitySet, true> bField(fieldName);

                std::vector<T> field(mesh.nPoints(), 0);

#ifdef MULTI_GPU

                const label_t nxBlocksPerGPU = (mesh.nxBlocks()) / mesh.nDevices<axis::X>(); // > Set to device::NUM_BLOCK_X
                const label_t nyBlocksPerGPU = (mesh.nyBlocks()) / mesh.nDevices<axis::Y>(); // > Set to device::NUM_BLOCK_Y
                const label_t nzBlocksPerGPU = (mesh.nzBlocks()) / mesh.nDevices<axis::Z>(); // > Set to device::NUM_BLOCK_Z

                // This is the loop we should be using for multi GPU, I think
                for (label_t GPU_z = 0; GPU_z < mesh.nDevices<axis::Z>(); GPU_z++)
                {
                    for (label_t GPU_y = 0; GPU_y < mesh.nDevices<axis::Y>(); GPU_y++)
                    {
                        for (label_t GPU_x = 0; GPU_x < mesh.nDevices<axis::X>(); GPU_x++)
                        {
                            // const label_t virtualDeviceIndex = GPU_x + GPU_y * nxGPUs + GPU_z * nxGPUs * nyGPUs;
                            // const label_t startIndex = virtualDeviceIndex * nPointsPerGPU;
                            // Fill this GPU's contiguous segment
                            grid_for(
                                nxBlocksPerGPU, nyBlocksPerGPU, nzBlocksPerGPU,
                                [&](const label_t bx, const label_t by, const label_t bz,
                                    const label_t tx, const label_t ty, const label_t tz)
                                {
                                    // Calculate global coordinates
                                    const label_t x = tx + block::nx() * (bx + (GPU_x * nxBlocksPerGPU));
                                    const label_t y = ty + block::ny() * (by + (GPU_y * nyBlocksPerGPU));
                                    const label_t z = tz + block::nz() * (bz + (GPU_z * nzBlocksPerGPU));

                                    // MODIFY FOR MULTI GPU
                                    const label_t index = host::idx(tx, ty, tz, bx, by, bz, nxBlocksPerGPU, nyBlocksPerGPU);

                                    const bool is_west = mesh.West(x);
                                    const bool is_east = mesh.East(x);
                                    const bool is_south = mesh.South(y);
                                    const bool is_north = mesh.North(y);
                                    const bool is_back = mesh.Back(z);
                                    const bool is_front = mesh.Front(z);

                                    const label_t boundary_count =
                                        static_cast<label_t>(is_west) +
                                        static_cast<label_t>(is_east) +
                                        static_cast<label_t>(is_south) +
                                        static_cast<label_t>(is_north) +
                                        static_cast<label_t>(is_back) +
                                        static_cast<label_t>(is_front);
                                    const T value_sum =
                                        (is_west * bField.West()) +
                                        (is_east * bField.East()) +
                                        (is_south * bField.South()) +
                                        (is_north * bField.North()) +
                                        (is_back * bField.Back()) +
                                        (is_front * bField.Front());

                                    field[index] = boundary_count > 0 ? value_sum / static_cast<T>(boundary_count) : bField.internalField();
                                });
                        }
                    }
                }
#else
                grid_for(
                    mesh.nxBlocks(), mesh.nyBlocks(), mesh.nzBlocks(),
                    [&](const label_t bx, const label_t by, const label_t bz,
                        const label_t tx, const label_t ty, const label_t tz)
                    {
                        const label_t x = (bx * block::nx()) + tx;
                        const label_t y = (by * block::ny()) + ty;
                        const label_t z = (bz * block::nz()) + tz;

                        const label_t index = host::idx(tx, ty, tz, bx, by, bz, mesh);

                        const bool is_west = mesh.West(x);
                        const bool is_east = mesh.East(x);
                        const bool is_south = mesh.South(y);
                        const bool is_north = mesh.North(y);
                        const bool is_back = mesh.Back(z);
                        const bool is_front = mesh.Front(z);

                        const label_t boundary_count =
                            static_cast<label_t>(is_west) +
                            static_cast<label_t>(is_east) +
                            static_cast<label_t>(is_south) +
                            static_cast<label_t>(is_north) +
                            static_cast<label_t>(is_back) +
                            static_cast<label_t>(is_front);
                        const T value_sum =
                            (is_west * bField.West()) +
                            (is_east * bField.East()) +
                            (is_south * bField.South()) +
                            (is_north * bField.North()) +
                            (is_back * bField.Back()) +
                            (is_front * bField.Front());

                        field[index] = boundary_count > 0 ? value_sum / static_cast<T>(boundary_count) : bField.internalField();
                    });
#endif
                return field;
            }
        };
    }
}

#endif

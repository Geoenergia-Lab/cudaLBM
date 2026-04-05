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
    This file defines the host array specialization that uses pageable memory
    (std::vector). This specialization is typically used for fields loaded from
    disk (checkpoints or initial fields) and provides initialisation from files
    or initial conditions.

Namespace
    LBM::host

SourceFiles
    host/paged.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_HOSTARRAY_PAGED_CUH
#define __MBLBM_HOSTARRAY_PAGED_CUH

namespace LBM
{
    namespace host
    {
        /**
         * @brief Host array using pageable memory (std::vector).
         *
         * This specialization stores data in a std::vector and provides
         * initialisation from files or initial conditions. It is typically used
         * for fields loaded from disk (checkpoints or initial fields).
         *
         * @tparam T Data type of array elements.
         * @tparam VelocitySet The velocity set (D3Q19 or D3Q27)
         * @tparam TimeType Type of time stepping (instantaneous or timeAverage)
         **/
        template <typename T, class VelocitySet, const time::type TimeType>
        class array<host::PAGED, T, VelocitySet, TimeType> : public arrayBase<T, VelocitySet, TimeType>
        {
            /**
             * @brief Bring base members into scope
             **/
            using arrayBase<T, VelocitySet, TimeType>::name_;
            using arrayBase<T, VelocitySet, TimeType>::mesh_;

        private:
            /**
             * @brief Alias for the current specialization
             **/
            using This = array<host::PAGED, T, VelocitySet, TimeType>;

        public:
            /**
             * @brief Construct a pageable array by reading from file or applying initial conditions.
             * @param[in] name Field name.
             * @param[in] mesh The lattice mesh
             * @param[in] programCtrl The program control object
             **/
            __host__ [[nodiscard]] array(
                const name_t &name,
                const host::latticeMesh &mesh,
                const programControl &programCtrl)
                : arrayBase<T, VelocitySet, TimeType>(name, mesh),
                  arr_(initialise_array(mesh, name, name, programCtrl)),
                  meanCount_(initialiseMeanCount(name, programCtrl)) {}

            __host__ [[nodiscard]] array(
                const name_t &name,
                const name_t &componentName,
                const host::latticeMesh &mesh,
                const programControl &programCtrl)
                : arrayBase<T, VelocitySet, TimeType>(name, mesh),
                  arr_(initialise_array(mesh, name, componentName, programCtrl)),
                  meanCount_(initialiseMeanCount(name, programCtrl)) {}

            /**
             * @brief Destructor
             **/
            __host__ ~array() {}

            /**
             * @brief Get const reference to the underlying vector.
             **/
            __host__ [[nodiscard]] inline constexpr const std::vector<T> &arr() const noexcept { return arr_; }

            /**
             * @brief Get raw pointer to the data (read‑only).
             **/
            __host__ [[nodiscard]] inline constexpr const T *data() const noexcept { return arr_.data(); }

            /**
             * @brief Element access (mutable).
             * @param[in] idx Index.
             * @return Reference to element.
             **/
            __host__ [[nodiscard]] inline constexpr T &operator[](const host::label_t idx) noexcept { return arr_[idx]; }

            /**
             * @brief Element access (read‑only).
             * @param[in] idx Index.
             * @return Const reference to element.
             **/
            __host__ [[nodiscard]] inline constexpr const T &operator[](const host::label_t idx) const noexcept { return arr_[idx]; }

            /**
             * @brief Get the number of elements.
             **/
            __host__ [[nodiscard]] inline constexpr host::label_t size() const noexcept { return arr_.size(); }

            /**
             * @brief Get the current averaging count (for time‑averaged fields).
             **/
            __host__ [[nodiscard]] inline constexpr host::label_t meanCount() const noexcept { return meanCount_; }

            /**
             * @brief Get a reference to the averaging count (for modification).
             **/
            __host__ [[nodiscard]] inline constexpr host::label_t &meanCountRef() noexcept { return meanCount_; }

        private:
            /**
             * @brief The actual data
             **/
            const std::vector<T> arr_;

            /**
             * @brief Number of averaged time steps
             **/
            host::label_t meanCount_;

            __host__ [[nodiscard]] static const std::vector<T> initialise_array(
                const host::latticeMesh &mesh,
                const name_t &fieldName,
                const name_t &componentName,
                const programControl &programCtrl)
            {
                if (!std::filesystem::is_directory("timeStep/" + std::to_string(programCtrl.latestTime())))
                {
                    std::cout << "Did not find directory timeStep/" << std::to_string(programCtrl.latestTime()) << std::endl;
                    return initialConditions(mesh, componentName);
                }
                else
                {
                    std::cout << "Reading field " << componentName << " from file " << fieldName << " for time step " << programCtrl.latestTime() << std::endl;

                    const name_t resolvedFileName = "timeStep/" + std::to_string(programCtrl.latestTime()) + "/" + fieldName + ".LBMBin";

                    return fileIO::readFieldByName<T>(resolvedFileName, componentName);
                }
            }

            /**
             * @brief Generate initial conditions with boundary handling.
             * @param[in] mesh The lattice mesh
             * @param[in] fieldName Field name (for boundary look‑up).
             * @return Vector containing the initial field.
             **/
            __host__ [[nodiscard]] static const std::vector<T> initialConditions(
                const host::latticeMesh &mesh,
                const name_t &fieldName)
            {
                const boundaryFields<VelocitySet, true> bField(fieldName);

                const host::label_t nxGPUs = mesh.nDevices<axis::X>();
                const host::label_t nyGPUs = mesh.nDevices<axis::Y>();
                const host::label_t nzGPUs = mesh.nDevices<axis::Z>();
                const host::label_t nPointsPerDevice = mesh.sizePerDevice();

                std::vector<T> field(mesh.size(), 0);

                const host::blockLabel nBlocksPerDevice = mesh.blocksPerDevice();

                GPU::forAll(
                    mesh.nDevices(),
                    [&](const host::label_t GPU_x, const host::label_t GPU_y, const host::label_t GPU_z)
                    {
                        const host::label_t virtualDeviceIndex = GPU::idx(GPU_x, GPU_y, GPU_z, mesh.nDevices<axis::X>(), mesh.nDevices<axis::Y>());

                        host::forAll(
                            nBlocksPerDevice,
                            [&](const host::label_t bx, const host::label_t by, const host::label_t bz,
                                const host::label_t tx, const host::label_t ty, const host::label_t tz)
                            {
                                // Global coordinates (for boundary detection)
                                const host::label_t x = tx + block::nx() * (bx + (GPU_x * nBlocksPerDevice.value<axis::X>()));
                                const host::label_t y = ty + block::ny() * (by + (GPU_y * nBlocksPerDevice.value<axis::Y>()));
                                const host::label_t z = tz + block::nz() * (bz + (GPU_z * nBlocksPerDevice.value<axis::Z>()));

                                // Local index within this GPU's segment
                                const host::label_t localIdx = host::idx(tx, ty, tz, bx, by, bz, nBlocksPerDevice.value<axis::X>(), nBlocksPerDevice.value<axis::Y>());

                                // Boundary detection
                                const bool is_west = mesh.West(x);
                                const bool is_east = mesh.East(x);
                                const bool is_south = mesh.South(y);
                                const bool is_north = mesh.North(y);
                                const bool is_back = mesh.Back(z);
                                const bool is_front = mesh.Front(z);

                                const host::label_t boundary_count =
                                    static_cast<host::label_t>(is_west) +
                                    static_cast<host::label_t>(is_east) +
                                    static_cast<host::label_t>(is_south) +
                                    static_cast<host::label_t>(is_north) +
                                    static_cast<host::label_t>(is_back) +
                                    static_cast<host::label_t>(is_front);

                                const T value_sum =
                                    (is_west * bField.West()) +
                                    (is_east * bField.East()) +
                                    (is_south * bField.South()) +
                                    (is_north * bField.North()) +
                                    (is_back * bField.Back()) +
                                    (is_front * bField.Front());

                                const T value = (boundary_count > 0) ? (value_sum / static_cast<T>(boundary_count)) : bField.internalField();

                                // Global index in host vector (per‑GPU segmented)
                                const host::label_t globalIdx = virtualDeviceIndex * nPointsPerDevice + localIdx;
                                field[globalIdx] = value;
                            });
                    });

                return field;
            }
        };
    }
}

#endif

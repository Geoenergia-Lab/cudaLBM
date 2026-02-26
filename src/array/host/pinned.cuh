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
    This file defines the host array specialization that uses pinned
    (page‑locked) memory. Pinned memory allows for faster data transfers
    between host and device, which can improve performance when copying
    large arrays. The class manages a raw pointer to pinned memory and
    provides methods for copying data from device pointers.

Namespace
    LBM::host

SourceFiles
    host/pinned.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_HOSTARRAY_PINNED_CUH
#define __MBLBM_HOSTARRAY_PINNED_CUH

namespace LBM
{
    namespace host
    {
        /**
         * @brief Host array using pinned (page‑locked) memory allocated with cudaMallocHost.
         *
         * This specialization manages a raw pointer to pinned memory and provides
         * methods for copying data from device pointers. The memory is automatically
         * freed in the destructor.
         *
         * @tparam T Data type of array elements.
         * @tparam VelocitySet The velocity set (D3Q19 or D3Q27)
         * @tparam TimeType Type of time stepping (instantaneous or timeAverage)
         **/
        template <typename T, class VelocitySet, const time::type TimeType>
        class array<host::PINNED, T, VelocitySet, TimeType> : public arrayBase<T, VelocitySet, TimeType>
        {
            /**
             * @brief Bring base members into scope
             **/
            using arrayBase<T, VelocitySet, TimeType>::name_;
            using arrayBase<T, VelocitySet, TimeType>::mesh_;

        public:
            /**
             * @brief Construct a pinned array of given size, zero‑initialised.
             * @param[in] nPoints Number of elements.
             * @param[in] mesh The lattice mesh
             **/
            __host__ [[nodiscard]] array(
                const label_t nPoints,
                const host::latticeMesh &mesh)
                : arrayBase<T, VelocitySet, TimeType>("", mesh),
                  ptr_(host::allocate<T>(nPoints, 0)),
                  nPoints_(nPoints) {}

            /**
             * @brief Construct a pinned array of given size, uniformly initialised to a value.
             * @param[in] nPoints Number of elements.
             * @param[in] val Initial value.
             * @param[in] mesh The lattice mesh
             **/
            __host__ [[nodiscard]] array(
                const label_t nPoints,
                const T val,
                const host::latticeMesh &mesh)
                : arrayBase<T, VelocitySet, TimeType>("", mesh),
                  ptr_(host::allocate<T>(nPoints, val)),
                  nPoints_(nPoints) {}

            /**
             * @brief Destructor – frees the pinned memory.
             **/
            __host__ ~array()
            {
                errorHandler::check(cudaFreeHost(const_cast<T *>(ptr_)));
            };

            /**
             * @brief Get raw pointer to the data (read‑only).
             **/
            __host__ [[nodiscard]] inline constexpr const T *data() const noexcept { return ptr_; }

            /**
             * @brief Get raw pointer to the data (mutable).
             **/
            __host__ [[nodiscard]] inline constexpr T *data() noexcept { return ptr_; }

            /**
             * @brief Element access (mutable).
             * @param[in] idx Index (0‑based).
             * @return Reference to element.
             **/
            __host__ [[nodiscard]] inline constexpr T &operator[](const label_t idx) noexcept
            {
                return ptr_[idx];
            }

            /**
             * @brief Element access (read‑only).
             * @param[in] idx Index (0‑based).
             * @return Const reference to element.
             **/
            __host__ [[nodiscard]] inline constexpr const T &operator[](const label_t idx) const noexcept
            {
                return ptr_[idx];
            }

            /**
             * @brief Get the number of elements.
             **/
            __host__ [[nodiscard]] inline constexpr label_t size() const noexcept { return nPoints_; }

            /**
             * @brief Copy data from a collection of device pointers into this array.
             *
             * The device pointers are assumed to point to a contiguous segment for one GPU.
             * The method copies each field's segment into the appropriate location in the
             * host array (which holds all fields contiguously: field0 + field1 + ...).
             *
             * @tparam N Number of fields (components).
             * @param[in] devPtrs Array of device pointers (one per field).
             * @param[in] mesh The lattice mesh
             * @param[in] virtualDeviceIndex Index of the GPU whose segment is being copied.
             **/
            template <const label_t N>
            __host__ void copy_from_device(
                const device::ptrCollection<N, T> &devPtrs,
                const host::latticeMesh &mesh,
                const label_t virtualDeviceIndex)
            {
                const label_t nxGPUs = mesh.nDevices<axis::X>();
                const label_t nyGPUs = mesh.nDevices<axis::Y>();
                const label_t nzGPUs = mesh.nDevices<axis::Z>();

                const label_t nxPointsPerDevice = mesh.dimension<axis::X>() / nxGPUs;
                const label_t nyPointsPerDevice = mesh.dimension<axis::Y>() / nyGPUs;
                const label_t nzPointsPerDevice = mesh.dimension<axis::Z>() / nzGPUs;
                const label_t nPointsPerDevice = nxPointsPerDevice * nyPointsPerDevice * nzPointsPerDevice;

                if (mesh.size() * N > nPoints_)
                {
                    throw std::runtime_error("Insufficient host array size");
                }

                for (label_t field = 0; field < N; ++field)
                {
                    errorHandler::check(cudaMemcpy(&(ptr_[(field * mesh.size()) + (virtualDeviceIndex * nPointsPerDevice)]), devPtrs[field], nPointsPerDevice * sizeof(T), cudaMemcpyDeviceToHost));
                }
            }

        private:
            /**
             * @brief Pointer to pinned host memory
             **/
            T *const ptrRestrict ptr_;

            /**
             * @brief Number of elements
             **/
            const label_t nPoints_;
        };
    }
}

#endif

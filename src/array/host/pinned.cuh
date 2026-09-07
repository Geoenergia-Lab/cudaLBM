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
         **/
        template <typename T>
        class array<host::PINNED, T> : public arrayBase<T>
        {
            /**
             * @brief Bring base members into scope
             **/
            using arrayBase<T>::mesh_;

        public:
            /**
             * @brief Construct a pinned array of given size, zero‑initialised.
             * @param[in] nPoints Number of elements.
             * @param[in] mesh The lattice mesh
             **/
            __host__ [[nodiscard]] array(
                const host::label_t nPoints,
                const host::latticeMesh &mesh)
                : arrayBase<T>("", mesh),
                  ptr_(host::allocate<T>(nPoints, 0)),
                  nPoints_(nPoints) {}

            /**
             * @brief Destructor - frees the pinned memory.
             **/
            __host__ ~array()
            {
                host::free(ptr_);
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
            __host__ [[nodiscard]] inline constexpr T &operator[](const host::label_t idx) noexcept
            {
                return ptr_[idx];
            }

            /**
             * @brief Element access (read‑only).
             * @param[in] idx Index (0‑based).
             * @return Const reference to element.
             **/
            __host__ [[nodiscard]] inline constexpr const T &operator[](const host::label_t idx) const noexcept
            {
                return ptr_[idx];
            }

            /**
             * @brief Get the number of elements.
             **/
            __host__ [[nodiscard]] inline constexpr host::label_t size() const noexcept { return nPoints_; }

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
            template <const host::label_t N>
            __host__ void copyFromDevice(
                const device::ptrCollection<N, const T> &devPtrs,
                const host::latticeMesh &mesh,
                const programControl &programCtrl,
                const host::label_t virtualDeviceIndex)
            {
                const host::label_t nxGPUs = mesh.nDevices<axis::X>();
                const host::label_t nyGPUs = mesh.nDevices<axis::Y>();
                const host::label_t nzGPUs = mesh.nDevices<axis::Z>();

                const host::label_t nxPointsPerDevice = mesh.dimension<axis::X>() / nxGPUs;
                const host::label_t nyPointsPerDevice = mesh.dimension<axis::Y>() / nyGPUs;
                const host::label_t nzPointsPerDevice = mesh.dimension<axis::Z>() / nzGPUs;
                const host::label_t nPointsPerDevice = nxPointsPerDevice * nyPointsPerDevice * nzPointsPerDevice;

                if (mesh.size() * N > nPoints_)
                {
                    throw std::runtime_error("Insufficient host array size");
                }

                for (host::label_t field = 0; field < N; field++)
                {
                    device::memcpyAsyncDeviceToHost(
                        &(ptr_[(field * mesh.size()) + (virtualDeviceIndex * nPointsPerDevice)]),
                        devPtrs[field],
                        nPointsPerDevice,
                        programCtrl.streams()[device::internalStreamID(virtualDeviceIndex)]);
                }
            }

        private:
            /**
             * @brief Pointer to pinned host memory
             **/
            hostPtr_t<T> ptr_;

            /**
             * @brief Number of elements
             **/
            const host::label_t nPoints_;
        };
    }
}

#endif

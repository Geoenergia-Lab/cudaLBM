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
    This file defines the primary template for device arrays, which is then
    specialised for different field types (skeleton and full fields). The
    primary template is not intended to be instantiated directly; instead, it
    serves as a base for the specializations that manage device memory for
    specific field types. The class provides common functionality for allocating
    and freeing device memory, as well as copying data between host and device.

Namespace
    LBM

SourceFiles
    device/array.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_DEVICEARRAY_CUH
#define __MBLBM_DEVICEARRAY_CUH

namespace LBM
{
    namespace device
    {
        /**
         * @brief Forward declaration of the primary template
         **/
        template <const field::type FieldType, typename T, class VelocitySet, const time::type TimeType>
        class array;

        /**
         * @brief Base class for device-side arrays, managing a collection of device pointers.
         *
         * This class holds the array of device pointers (one per GPU), along with references
         * to the mesh and program control. It provides a virtual destructor that automatically
         * frees all associated device memory.
         *
         * @tparam T Fundamental type of the array elements.
         **/
        template <typename T>
        class arrayBase
        {
            static_assert(MULTI_GPU_ASSERTION(), MULTI_GPU_MSG_NOTE(device::arrayBase, "Need to ensure that device::arrayBase and host::arrayBase check the device decomposition of the file from which they read; this is not always going to be the same as the decomposition specified in the deviceDecomposition file."));

        protected:
            /**
             * @brief Array of device pointers (one per GPU)
             **/
            T **ptr_;

            /**
             * @brief Reference to the lattice mesh
             **/
            const host::latticeMesh &mesh_;

            /**
             * @brief Reference to program control
             **/
            const programControl &programCtrl_;

            /**
             * @brief Construct a base object without initialising the pointer array.
             * @param[in] mesh The lattice mesh
             * @param[in] programCtrl The program control object
             **/
            __host__ [[nodiscard]] arrayBase(
                const host::latticeMesh &mesh,
                const programControl &programCtrl) noexcept
                : ptr_(nullptr),
                  mesh_(mesh),
                  programCtrl_(programCtrl) {}

            /**
             * @brief Construct a base object with an already allocated pointer array.
             * @param[in] ptr Pointer to the array of device pointers (host memory).
             * @param[in] mesh The lattice mesh
             * @param[in] programCtrl The program control object
             **/
            __host__ [[nodiscard]] arrayBase(
                T **ptr,
                const host::latticeMesh &mesh,
                const programControl &programCtrl) noexcept
                : ptr_(ptr),
                  mesh_(mesh),
                  programCtrl_(programCtrl) {}

            /**
             * @brief Allocate and copy one GPU segment of the skeleton array.
             * @tparam alpha Axis direction (used to compute number of faces).
             * @param[in] mesh The lattice mesh
             * @param[in] hostArrayGlobal Pointer to full host array.
             * @param[in] GPU_x, GPU_y, GPU_z GPU grid coordinates.
             * @param[in] programCtrl The program control object
             * @param[in] allocationSize Number of points allocated per GPU (used to compute segment size).
             * @return Device pointer for the segment.
             **/
            __host__ [[nodiscard]] static T *allocate_device_segment(
                const host::latticeMesh &mesh,
                const T *hostArrayGlobal,
                const host::label_t GPU_x,
                const host::label_t GPU_y,
                const host::label_t GPU_z,
                const programControl &programCtrl,
                const host::label_t allocationSize)
            {
                const host::label_t virtualDeviceIndex = GPU::idx(GPU_x, GPU_y, GPU_z, mesh.nDevices<axis::X>(), mesh.nDevices<axis::Y>());
                const host::label_t startIndex = virtualDeviceIndex * allocationSize;

                T *devPtr = device::allocate<T>(allocationSize, programCtrl.deviceList()[virtualDeviceIndex]);

                device::copy(devPtr, &(hostArrayGlobal[startIndex]), allocationSize, programCtrl.deviceList()[virtualDeviceIndex]);

                return devPtr;
            }

            /**
             * @brief Allocate all GPU segments for a skeleton array from a raw host pointer.
             * @tparam alpha Axis direction.
             * @param[in] mesh The lattice mesh
             * @param[in] hostArrayGlobal Raw pointer to host data.
             * @param[in] programCtrl The program control object
             * @return Host array of device pointers (one per GPU).
             **/
            __host__ [[nodiscard]] static T **allocate_on_devices(
                const host::latticeMesh &mesh,
                const T *hostArrayGlobal,
                const programControl &programCtrl,
                const host::label_t allocationSize)
            {
                T **hostPtrsToDevice = host::allocate<T *>(mesh.nDevices().size(), nullptr);

                GPU::forAll(
                    mesh.nDevices(),
                    [&](const host::label_t GPU_x, const host::label_t GPU_y, const host::label_t GPU_z)
                    {
                        const host::label_t virtualDeviceIndex = GPU::idx(GPU_x, GPU_y, GPU_z, mesh.nDevices<axis::X>(), mesh.nDevices<axis::Y>());
                        hostPtrsToDevice[virtualDeviceIndex] = allocate_device_segment(mesh, hostArrayGlobal, GPU_x, GPU_y, GPU_z, programCtrl, allocationSize);
                    });

                return hostPtrsToDevice;
            }

        public:
            /**
             * @brief Virtual destructor – automatically releases all device memory.
             **/
            __host__ virtual ~arrayBase()
            {
                free_device_pointers();
            }

            /**
             * @brief Disable copying
             **/
            __host__ [[nodiscard]] arrayBase(const arrayBase &) = delete;
            __host__ [[nodiscard]] arrayBase &operator=(const arrayBase &) = delete;

        private:
            /**
             * @brief Free all device pointers and the host-side pointer array.
             **/
            __host__ void free_device_pointers() noexcept
            {
                errorHandler::check(cudaDeviceSynchronize());

                if (ptr_ == nullptr)
                {
                    return;
                }

                GPU::forAll(
                    mesh_.nDevices(),
                    [&](host::label_t GPU_x, host::label_t GPU_y, host::label_t GPU_z)
                    {
                        const host::label_t virtualDeviceIndex = GPU::idx(GPU_x, GPU_y, GPU_z, mesh_.nDevices<axis::X>(), mesh_.nDevices<axis::Y>());
                        if (ptr_[virtualDeviceIndex] != nullptr)
                        {
                            errorHandler::check(cudaDeviceSynchronize());
                            errorHandler::check(cudaSetDevice(programCtrl_.deviceList()[virtualDeviceIndex]));
                            errorHandler::check(cudaDeviceSynchronize());
                            errorHandler::check(cudaFree(const_cast<T *>(ptr_[virtualDeviceIndex])));
                            errorHandler::check(cudaDeviceSynchronize());
                        }
                    });

                errorHandler::check(cudaDeviceSynchronize());
                errorHandler::check(cudaSetDevice(programCtrl_.deviceList()[0]));
                errorHandler::check(cudaDeviceSynchronize());
                errorHandler::check(cudaFreeHost(const_cast<T **>(ptr_)));
                errorHandler::check(cudaDeviceSynchronize());
            }
        };
    }
}

#include "skeleton.cuh"
#include "fullField.cuh"

#endif

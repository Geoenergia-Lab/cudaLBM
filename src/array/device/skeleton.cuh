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
    This file defines the device array specialization for skeleton fields. The
    skeleton field is a special type of field that does not carry a name or
    time-averaging information and is typically used for inter-block halos on
    the device. This specialization manages device memory pointers and provides
    methods for accessing and manipulating the data on the GPU.

Namespace
    LBM

SourceFiles
    device/skeleton.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_DEVICEARRAY_SKELETON_CUH
#define __MBLBM_DEVICEARRAY_SKELETON_CUH

namespace LBM
{
    namespace device
    {
        /**
         * @brief Device array for skeleton fields (no name, no time averaging).
         *
         * This specialization stores only a pointer to device memory and does not
         * carry field name or time‑averaging information. It is typically used for
         * intermediate or temporary data.
         *
         * @tparam T Fundamental type of the array.
         * @tparam VelocitySet The velocity set (D3Q19 or D3Q27)
         **/
        template <typename T, class VelocitySet>
        class array<field::SKELETON, T, VelocitySet, time::instantaneous> : public arrayBase<T>
        {
        private:
            /**
             * @brief Bring base members into scope
             **/
            using arrayBase<T>::ptr_;
            using arrayBase<T>::allocate_device_segment;

            /**
             * @brief Alias for the current specialization
             **/
            using This = array<field::SKELETON, T, VelocitySet, time::instantaneous>;

        public:
            /**
             * @brief Construct a skeleton device array from host data.
             * @tparam Alpha Axis along which the array is defined (for decomposition).
             * @param[in] hostArray Source data on the host.
             * @param[in] mesh The lattice mesh
             * @param[in] programCtrl The program control object
             * @param[in] alpha Integral constant indicating the axis.
             **/
            template <const axis::type Alpha>
            __host__ [[nodiscard]] array(
                const std::vector<T> &hostArray,
                const host::latticeMesh &mesh,
                const programControl &programCtrl,
                const integralConstant<axis::type, Alpha> &alpha)
                : arrayBase<T>(
                      This::template allocate_on_devices<alpha.value>(
                          mesh,
                          hostArray,
                          programCtrl),
                      mesh,
                      programCtrl)
            {
                static_assert(MULTI_GPU_ASSERTION(), MULTI_GPU_MSG_NOTE(device::array::<field::SKELETON>, "Need to decompose skeleton amongst devices"));
            }

            /**
             * @brief Get read-only pointer to device memory for a given GPU.
             * @param[in] i Index of the GPU (virtual device index).
             * @return Const pointer to device memory.
             **/
            __device__ __host__ [[nodiscard]] inline const T *constPtr(const label_t i) const noexcept
            {
                return ptr_[i];
            }

            /**
             * @brief Get mutable pointer to device memory for a given GPU.
             * @param[in] i Index of the GPU (virtual device index).
             * @return Pointer to device memory.
             **/
            __device__ __host__ [[nodiscard]] inline T *ptr(const label_t i) noexcept
            {
                return ptr_[i];
            }

            /**
             * @brief Provide a reference to the device pointer for swapping operations.
             * @param[in] i Index of the GPU (virtual device index).
             * @return Reference to the pointer (host side).
             **/
            __host__ [[nodiscard]] inline constexpr T * ptrRestrict & ptrRef(const label_t i) noexcept
            {
                return ptr_[i];
            }

        private:
            __host__ [[nodiscard]] static T *allocate_halo_segment(
                [[maybe_unused]] const host::latticeMesh &mesh,
                [[maybe_unused]] const T *hostArrayGlobal,
                [[maybe_unused]] const label_t GPU_x,
                [[maybe_unused]] const label_t GPU_y,
                [[maybe_unused]] const label_t GPU_z,
                [[maybe_unused]] const programControl &programCtrl,
                [[maybe_unused]] const label_t allocationSize)
            {
                const label_t virtualDeviceIndex = GPU::idx(GPU_x, GPU_y, GPU_z, mesh.nDevices<axis::X>(), mesh.nDevices<axis::Y>());
                const label_t startIndex = virtualDeviceIndex * allocationSize;

                T *devPtr = device::allocate<T>(allocationSize, programCtrl.deviceList()[virtualDeviceIndex]);

                device::copy(devPtr, &(hostArrayGlobal[startIndex]), allocationSize, programCtrl.deviceList()[virtualDeviceIndex]);

                return devPtr;
            }

            __host__ [[nodiscard]] static T **allocate_halo_on_devices(
                [[maybe_unused]] const host::latticeMesh &mesh,
                [[maybe_unused]] const T *hostArrayGlobal,
                [[maybe_unused]] const programControl &programCtrl,
                [[maybe_unused]] const label_t allocationSize)
            {
                T **hostPtrsToDevice = host::allocate<T *>(mesh.nDevices().size(), nullptr);

                GPU::forAll(
                    mesh.nDevices(),
                    [&](label_t GPU_x, label_t GPU_y, label_t GPU_z)
                    {
                        std::cout << "deviceID {" << GPU_x << " " << GPU_y << " " << GPU_z << "}:" << std::endl;
                        std::cout << "East boundary: " << (GPU_x < mesh.nDevices<axis::X>() - 1) << std::endl;
                        std::cout << "West boundary: " << (GPU_x > 0) << std::endl;
                        std::cout << "North boundary: " << (GPU_y < mesh.nDevices<axis::Y>() - 1) << std::endl;
                        std::cout << "South boundary: " << (GPU_y > 0) << std::endl;
                        std::cout << "Back boundary: " << (GPU_z < mesh.nDevices<axis::Z>() - 1) << std::endl;
                        std::cout << "Front boundary: " << (GPU_z > 0) << std::endl;

                        // const label_t virtualDeviceIndex = GPU::idx(GPU_x, GPU_y, GPU_z, mesh.nDevices<axis::X>(), mesh.nDevices<axis::Y>());
                        // hostPtrsToDevice[virtualDeviceIndex] = allocate_halo_segment(mesh, hostArrayGlobal, GPU_x, GPU_y, GPU_z, programCtrl, allocationSize);
                    });

                return hostPtrsToDevice;
            }

            /**
             * @brief Allocate all GPU segments for a skeleton array from a std::vector.
             * @tparam alpha Axis direction.
             * @param[in] mesh The lattice mesh
             * @param[in] hostArrayGlobal Source vector.
             * @param[in] programCtrl The program control object
             * @return Host array of device pointers.
             **/
            template <const axis::type alpha>
            __host__ [[nodiscard]] static T **allocate_on_devices(
                const host::latticeMesh &mesh,
                const std::vector<T> &hostArrayGlobal,
                const programControl &programCtrl)
            {
                if constexpr (false)
                {
                    return allocate_halo_on_devices(mesh, hostArrayGlobal.data(), programCtrl, mesh.nFacesPerDevice<alpha, VelocitySet::QF()>());
                }
                else
                {
                    return arrayBase<T>::allocate_on_devices(mesh, hostArrayGlobal.data(), programCtrl, mesh.nFacesPerDevice<alpha, VelocitySet::QF()>());
                }
            }
        };
    }
}

#endif

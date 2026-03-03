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
            template <const axis::type Alpha, const int Coeff>
            __host__ [[nodiscard]] array(
                const std::vector<T> &hostArray,
                const host::latticeMesh &mesh,
                const programControl &programCtrl,
                const integralConstant<axis::type, Alpha> &alpha,
                const integralConstant<int, Coeff> &coeff)
                : arrayBase<T>(
                      This::template allocate_on_devices<alpha.value, coeff.value>(
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
            template <const axis::type alpha, const label_t QF>
            __host__ [[nodiscard]] static inline label_t idxPopTest(
                const label_t pop,
                const blockLabel_t &Tx,
                const blockLabel_t &Bx,
                const label_t nxBlocks,
                const label_t nyBlocks) noexcept
            {
                return Tx.value<axis::orthogonal<alpha, 0>()>() + block::n<axis::orthogonal<alpha, 0>()>() * (Tx.value<axis::orthogonal<alpha, 1>()>() + block::n<axis::orthogonal<alpha, 1>()>() * (pop + QF * (Bx.x + nxBlocks * (Bx.y + nyBlocks * Bx.z))));
            }

            template <const axis::type alpha>
            __host__ [[nodiscard]] static inline constexpr std::size_t AllocationSize(
                const label_t GPU_x,
                const label_t GPU_y,
                const label_t GPU_z,
                const host::latticeMesh &mesh) noexcept
            {
                const std::size_t EastBoundary = static_cast<std::size_t>(GPU_x < mesh.nDevices<axis::X>() - 1);
                const std::size_t WestBoundary = static_cast<std::size_t>(GPU_x > 0);
                const std::size_t NorthBoundary = static_cast<std::size_t>(GPU_y < mesh.nDevices<axis::Y>() - 1);
                const std::size_t SouthBoundary = static_cast<std::size_t>(GPU_y > 0);
                const std::size_t FrontBoundary = static_cast<std::size_t>(GPU_z < mesh.nDevices<axis::Z>() - 1);
                const std::size_t BackBoundary = static_cast<std::size_t>(GPU_z > 0);

                if constexpr (alpha == axis::X)
                {
                    return mesh.nFacesPerDevice<alpha, VelocitySet::QF(), std::size_t>(EastBoundary, WestBoundary);
                }

                if constexpr (alpha == axis::Y)
                {
                    return mesh.nFacesPerDevice<alpha, VelocitySet::QF(), std::size_t>(NorthBoundary, SouthBoundary);
                }

                if constexpr (alpha == axis::Z)
                {
                    return mesh.nFacesPerDevice<alpha, VelocitySet::QF(), std::size_t>(BackBoundary, FrontBoundary);
                }
            }

            template <const axis::type alpha, const int coeff>
            __host__ [[nodiscard]] static T **allocate_halo_on_devices(
                const host::latticeMesh &mesh,
                const T *hostArrayGlobal,
                const programControl &programCtrl,
                const std::size_t globalSize)
            {
                T **hostPtrsToDevice = host::allocate<T *>(mesh.nDevices().size(), nullptr);

                // Special case if we have only 1 GPU since we do not need to decompose
                if ((mesh.nDevices<axis::X>() == 1) && (mesh.nDevices<axis::Y>() == 1) && (mesh.nDevices<axis::Z>() == 1))
                {
                    // We can just go straight to the allocation and copy step
                    const label_t virtualDeviceIndex = GPU::idx(static_cast<label_t>(0), static_cast<label_t>(0), static_cast<label_t>(0), mesh.nDevices<axis::X>(), mesh.nDevices<axis::Y>());
                    hostPtrsToDevice[virtualDeviceIndex] = allocate_halo_segment(hostArrayGlobal, virtualDeviceIndex, programCtrl, globalSize);
                }
                else
                {
                    GPU::forAll(
                        mesh.nDevices(),
                        [&](label_t GPU_x, label_t GPU_y, label_t GPU_z)
                        {
                            const label_t EastBoundary = static_cast<label_t>(GPU_x < mesh.nDevices<axis::X>() - 1);
                            const label_t WestBoundary = static_cast<label_t>(GPU_x > 0);
                            const label_t NorthBoundary = static_cast<label_t>(GPU_y < mesh.nDevices<axis::Y>() - 1);
                            const label_t SouthBoundary = static_cast<label_t>(GPU_y > 0);
                            const label_t FrontBoundary = static_cast<label_t>(GPU_z < mesh.nDevices<axis::Z>() - 1);
                            const label_t BackBoundary = static_cast<label_t>(GPU_z > 0);
                            std::cout << "deviceID {" << GPU_x << " " << GPU_y << " " << GPU_z << "}:" << std::endl;
                            std::cout << "West boundary: " << (WestBoundary ? "true" : "false") << std::endl;
                            std::cout << "East boundary: " << (EastBoundary ? "true" : "false") << std::endl;
                            std::cout << "South boundary: " << (SouthBoundary ? "true" : "false") << std::endl;
                            std::cout << "North boundary: " << (NorthBoundary ? "true" : "false") << std::endl;
                            std::cout << "Back boundary: " << (BackBoundary ? "true" : "false") << std::endl;
                            std::cout << "Front boundary: " << (FrontBoundary ? "true" : "false") << std::endl;
                            std::cout << std::endl;

                            device::copyToSymbol(device::STREAMING_OFFSET_WEST, WestBoundary);
                            device::copyToSymbol(device::STREAMING_OFFSET_EAST, EastBoundary);
                            device::copyToSymbol(device::STREAMING_OFFSET_SOUTH, SouthBoundary);
                            device::copyToSymbol(device::STREAMING_OFFSET_NORTH, NorthBoundary);
                            device::copyToSymbol(device::STREAMING_OFFSET_BACK, BackBoundary);
                            device::copyToSymbol(device::STREAMING_OFFSET_FRONT, FrontBoundary);

                            // Now, we should create a std vector containing the halo for this device
                            // The halo should begin at bx = (gpu_x * nxBlocksPerGPU) - WestBoundary, I think
                            // Same for the other boundaries
                            const std::size_t partitionAllocationSize = AllocationSize<alpha>(GPU_x, GPU_y, GPU_z, mesh);

                            std::vector<scalar_t> haloAlloc(partitionAllocationSize, 0);
                            const label_t nxBlocksPerDevice = mesh.blocksPerDevice<axis::X>() + WestBoundary + EastBoundary;
                            const label_t nyBlocksPerDevice = mesh.blocksPerDevice<axis::Y>() + SouthBoundary + NorthBoundary;
                            const label_t nzBlocksPerDevice = mesh.blocksPerDevice<axis::Z>() + BackBoundary + FrontBoundary;

                            // Calculate the first x,y,z block indices
                            const label_t bx0 = (GPU_x * nxBlocksPerDevice) - WestBoundary;
                            const label_t by0 = (GPU_y * nyBlocksPerDevice) - SouthBoundary;
                            const label_t bz0 = (GPU_z * nzBlocksPerDevice) - BackBoundary;

                            // This is the correct loop structure.
                            // Loop over the blocks: make sure to add the offsets and make these per GPU
                            for (label_t bz = 0; bz < nzBlocksPerDevice; bz++)
                            {
                                for (label_t by = 0; by < nyBlocksPerDevice; by++)
                                {
                                    for (label_t bx = 0; bx < nxBlocksPerDevice; bx++)
                                    {
                                        for (label_t i = 0; i < VelocitySet::QF(); i++)
                                        {
                                            // Loop over the threads that are perpendicular to alpha
                                            // These inner nested loops are correct, I believe
                                            // Second perpendicular axis
                                            for (label_t tb = 0; tb < block::n<axis::orthogonal<alpha, 1>()>(); tb++)
                                            {
                                                // First perpendicular axis
                                                for (label_t ta = 0; ta < block::n<axis::orthogonal<alpha, 0>()>(); ta++)
                                                {
                                                    // Get the 3d indices on the face
                                                    const blockLabel_t Tx = axis::to_3d<alpha, coeff>(ta, tb);

                                                    const blockLabel_t Bx(bx, by, bz);
                                                    const blockLabel_t Bx_global(bx0 + bx, by0 + by, bz0 + bz);

                                                    const label_t global_idx = (idxPopTest<alpha, VelocitySet::QF()>(i, Tx, Bx_global, mesh.nBlocks<axis::X>(), mesh.nBlocks<axis::Y>())) % (static_cast<label_t>(globalSize));

                                                    // Local index in haloAlloc (same layout, but using local block dimensions)
                                                    const label_t local_idx = (idxPopTest<alpha, VelocitySet::QF()>(i, Tx, Bx, nxBlocksPerDevice, nyBlocksPerDevice)) % (static_cast<label_t>(partitionAllocationSize));

                                                    if (static_cast<std::size_t>(global_idx) > globalSize)
                                                    {
                                                        std::cout << "Access error in hostArrayGlobal: " << global_idx << std::endl;
                                                        std::cout << "local_idx: " << local_idx << std::endl;
                                                        std::cout << "deviceID:" << std::endl;
                                                        std::cout << GPU_x << ", " << GPU_y << ", " << GPU_z << std::endl;
                                                        std::cout << nxBlocksPerDevice << " x blocks per device" << std::endl;
                                                        std::cout << nyBlocksPerDevice << " y blocks per device" << std::endl;
                                                        std::cout << nzBlocksPerDevice << " z blocks per device" << std::endl;
                                                        Tx.print("Tx");
                                                        Bx_global.print("BxGlobal");
                                                        throw std::runtime_error("Access error in hostArrayGlobal");
                                                    }

                                                    if (local_idx > haloAlloc.size())
                                                    {
                                                        std::cout << "Access error in haloAlloc" << std::endl;
                                                        throw std::runtime_error("Access error in haloAlloc");
                                                    }

                                                    haloAlloc[local_idx] = hostArrayGlobal[global_idx];
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            const label_t virtualDeviceIndex = GPU::idx(GPU_x, GPU_y, GPU_z, mesh.nDevices<axis::X>(), mesh.nDevices<axis::Y>());
                            hostPtrsToDevice[virtualDeviceIndex] = allocate_halo_segment(haloAlloc.data(), virtualDeviceIndex, programCtrl, partitionAllocationSize);
                        });
                }

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
            template <const axis::type alpha, const int coeff>
            __host__ [[nodiscard]] static T **allocate_on_devices(
                const host::latticeMesh &mesh,
                const std::vector<T> &hostArrayGlobal,
                const programControl &programCtrl)
            {
                if constexpr (true)
                {
                    return allocate_halo_on_devices<alpha, coeff>(mesh, hostArrayGlobal.data(), programCtrl, hostArrayGlobal.size());
                }
                else
                {
                    return arrayBase<T>::allocate_on_devices(mesh, hostArrayGlobal.data(), programCtrl, mesh.nFacesPerDevice<alpha, VelocitySet::QF()>());
                }
            }

            __host__ [[nodiscard]] static T *allocate_halo_segment(
                const T *hostArrayGlobal,
                const label_t virtualDeviceIndex,
                const programControl &programCtrl,
                const std::size_t partitionAllocationSize)
            {
                T *devPtr = device::allocate<T>(partitionAllocationSize, programCtrl.deviceList()[virtualDeviceIndex]);

                device::copy(devPtr, hostArrayGlobal, partitionAllocationSize, programCtrl.deviceList()[virtualDeviceIndex]);

                return devPtr;
            }
        };
    }
}

#endif

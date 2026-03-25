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
                      programCtrl) {}

            /**
             * @brief Get read-only pointer to device memory for a given GPU.
             * @param[in] i Index of the GPU (virtual device index).
             * @return Const pointer to device memory.
             **/
            __device__ __host__ [[nodiscard]] inline const T *constPtr(const host::label_t i) const noexcept
            {
                return ptr_[i];
            }

            /**
             * @brief Get mutable pointer to device memory for a given GPU.
             * @param[in] i Index of the GPU (virtual device index).
             * @return Pointer to device memory.
             **/
            __device__ __host__ [[nodiscard]] inline T *ptr(const host::label_t i) noexcept
            {
                return ptr_[i];
            }

            /**
             * @brief Provide a reference to the device pointer for swapping operations.
             * @param[in] i Index of the GPU (virtual device index).
             * @return Reference to the pointer (host side).
             **/
            __host__ [[nodiscard]] inline constexpr T * ptrRestrict & ptrRef(const host::label_t i) noexcept
            {
                return ptr_[i];
            }

        private:
            template <const axis::type alpha, const host::label_t QF>
            __host__ [[nodiscard]] static inline host::label_t idxPopTest(
                const host::label_t pop,
                const host::threadLabel &Tx,
                const host::blockLabel &Bx,
                const host::label_t nxBlocks,
                const host::label_t nyBlocks) noexcept
            {
                return Tx.value<axis::orthogonal<alpha, 0>()>() + block::n<axis::orthogonal<alpha, 0>()>() * (Tx.value<axis::orthogonal<alpha, 1>()>() + block::n<axis::orthogonal<alpha, 1>()>() * (pop + QF * (Bx.x + nxBlocks * (Bx.y + nyBlocks * Bx.z))));
            }

            template <const axis::type alpha>
            __host__ [[nodiscard]] static inline constexpr host::label_t AllocationSize(
                const host::label_t nxBlocksTrue,
                const host::label_t nyBlocksTrue,
                const host::label_t nzBlocksTrue) noexcept
            {
                return VelocitySet::template QF<host::label_t>() * ((static_cast<host::label_t>(nxBlocksTrue) * static_cast<host::label_t>(nyBlocksTrue) * static_cast<host::label_t>(nzBlocksTrue) * block::nx<host::label_t>() * block::ny<host::label_t>() * block::nz<host::label_t>()) / block::n<alpha, host::label_t>());
            }

            template <const axis::type alpha>
            __host__ [[nodiscard]] static inline consteval const char *axisName() noexcept
            {
                if constexpr (alpha == axis::X)
                {
                    return "X";
                }

                if constexpr (alpha == axis::Y)
                {
                    return "Y";
                }

                if constexpr (alpha == axis::Z)
                {
                    return "Z";
                }
            }

            template <const axis::type alpha, const int coeff>
            __host__ [[nodiscard]] static const std::vector<scalar_t> halo_partition(
                const host::latticeMesh &mesh,
                const T *hostArrayGlobal,
                const host::label_t GPU_x,
                const host::label_t GPU_y,
                const host::label_t GPU_z)
            {
                // Get the number of non-halo blocks per device
                const host::label_t nxBlocksPerGPU = mesh.blocksPerDevice<axis::X>();
                const host::label_t nyBlocksPerGPU = mesh.blocksPerDevice<axis::Y>();
                const host::label_t nzBlocksPerGPU = mesh.blocksPerDevice<axis::Z>();

                // So, the Z allocation size is nz blocks + haloHasExtraFace
                const host::label_t nxBlocksTrue = nxBlocksPerGPU;
                const host::label_t nyBlocksTrue = nyBlocksPerGPU;
                const host::label_t nzBlocksTrue = nzBlocksPerGPU;

                // Get the starting block indices for this GPU
                // These indices are not offset to account for the extra halo
                const host::label_t bx0 = GPU_x * nxBlocksPerGPU;
                const host::label_t by0 = GPU_y * nyBlocksPerGPU;
                const host::label_t bz0 = GPU_z * nzBlocksPerGPU;

                const host::label_t partitionAllocationSize = AllocationSize<alpha>(nxBlocksTrue, nyBlocksTrue, nzBlocksTrue);
                std::vector<scalar_t> haloAlloc(partitionAllocationSize, 0);

                for (host::label_t bz = 0; bz < nzBlocksTrue; bz++)
                {
                    for (host::label_t by = 0; by < nyBlocksTrue; by++)
                    {
                        for (host::label_t bx = 0; bx < nxBlocksTrue; bx++)
                        {
                            // Second perpendicular axis
                            for (host::label_t tb = 0; tb < block::n<axis::orthogonal<alpha, 1>()>(); tb++)
                            {
                                // First perpendicular axis
                                for (host::label_t ta = 0; ta < block::n<axis::orthogonal<alpha, 0>()>(); ta++)
                                {
                                    // Get the 3d indices on the face
                                    // Note: This call might not be correct, I am not sure if it should be + or - coeff
                                    // I think it should actually be + coeff, since we want to pull from the edge of the block propagating to the current
                                    const host::blockLabel Tx = axis::to_3d<alpha, coeff>(ta, tb);

                                    // The block label in the segment of the halo
                                    const host::blockLabel Bx(bx, by, bz);

                                    // The block label in the global halo
                                    const host::blockLabel Bx_global(bx0 + bx, by0 + by, bz0 + bz);

                                    // Local index in haloAlloc (same layout, but using local block dimensions)
                                    for (host::label_t i = 0; i < VelocitySet::QF(); i++)
                                    {
                                        // Linear index in the global matrix
                                        const host::label_t global_idx = idxPopTest<alpha, VelocitySet::QF()>(i, Tx, Bx_global, mesh.nBlocks<axis::X>(), mesh.nBlocks<axis::Y>());

                                        const host::label_t local_idx = idxPopTest<alpha, VelocitySet::QF()>(i, Tx, Bx, nxBlocksTrue, nyBlocksTrue);

                                        haloAlloc[local_idx] = hostArrayGlobal[global_idx];
                                    }
                                }
                            }
                        }
                    }
                }

                return haloAlloc;
            }

            template <const axis::type alpha, const int coeff>
            __host__ [[nodiscard]] static T **allocate_halo_on_devices(
                const host::latticeMesh &mesh,
                const T *hostArrayGlobal,
                const programControl &programCtrl,
                const host::label_t globalSize)
            {
                T **hostPtrsToDevice = host::allocate<T *>(mesh.nDevices().size(), nullptr);

                // Special case if we have only 1 GPU since we do not need to decompose
                if ((mesh.nDevices<axis::X>() == 1) && (mesh.nDevices<axis::Y>() == 1) && (mesh.nDevices<axis::Z>() == 1))
                {
                    // We can just go straight to the allocation and copy step
                    const host::label_t virtualDeviceIndex = GPU::idx(static_cast<device::label_t>(0), static_cast<device::label_t>(0), static_cast<device::label_t>(0), mesh.nDevices<axis::X>(), mesh.nDevices<axis::Y>());
                    hostPtrsToDevice[virtualDeviceIndex] = allocate_halo_segment(hostArrayGlobal, virtualDeviceIndex, programCtrl, globalSize);
                }
                else
                {
                    GPU::forAll(
                        mesh.nDevices(),
                        [&](const host::label_t GPU_x, const host::label_t GPU_y, const host::label_t GPU_z)
                        {
                            const host::label_t virtualDeviceIndex = GPU::idx(GPU_x, GPU_y, GPU_z, mesh.nDevices<axis::X>(), mesh.nDevices<axis::Y>());

                            errorHandler::check(cudaDeviceSynchronize());
                            errorHandler::check(cudaSetDevice(programCtrl.deviceList()[virtualDeviceIndex]));
                            errorHandler::check(cudaDeviceSynchronize());

                            // Get the number of non-halo blocks per device
                            const std::vector<scalar_t> haloAlloc = halo_partition<alpha, coeff>(mesh, hostArrayGlobal, GPU_x, GPU_y, GPU_z);

                            hostPtrsToDevice[virtualDeviceIndex] = allocate_halo_segment(haloAlloc.data(), virtualDeviceIndex, programCtrl, haloAlloc.size());
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
                return allocate_halo_on_devices<alpha, coeff>(mesh, hostArrayGlobal.data(), programCtrl, hostArrayGlobal.size());
            }

            __host__ [[nodiscard]] static T *allocate_halo_segment(
                const T *hostArrayGlobal,
                const host::label_t virtualDeviceIndex,
                const programControl &programCtrl,
                const host::label_t partitionAllocationSize)
            {
                T *devPtr = device::allocate<T>(partitionAllocationSize, programCtrl.deviceList()[virtualDeviceIndex]);

                device::copy(devPtr, hostArrayGlobal, partitionAllocationSize, programCtrl.deviceList()[virtualDeviceIndex]);

                return devPtr;
            }
        };
    }
}

#endif

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
    A templated class for allocating arrays on the GPU

Namespace
    LBM::device

SourceFiles
    deviceArray.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_DEVICEARRAY_CUH
#define __MBLBM_DEVICEARRAY_CUH

namespace LBM
{
    namespace device
    {
        template <const field::type FieldType, typename T, class VelocitySet, const time::type TimeType>
        class array;

        /**
         * @class Array holding only a pointer - no name or mesh information
         * @tparam T Fundamental type of the array
         * @tparam VelocitySet The velocity set
         * @tparam TimeType Type of time stepping (instantaneous or time-averaged)
         **/
        template <typename T, class VelocitySet, const time::type TimeType>
        class array<field::SKELETON, T, VelocitySet, TimeType>
        {
        public:
            __host__ [[nodiscard]] array(const std::vector<T> &hostArray)
                : ptr_(device::allocateArray<T>(hostArray))
            {
                static_assert(MULTI_GPU_ASSERTION(), MULTI_GPU_MSG_NOTE(device::array::<field::SKELETON>, "Need to decompose skeleton amongst devices"));
            };

            /**
             * @brief Destructor - automatically releases device memory
             * @note Noexcept guarantee: failsafe if cudaFree fails
             **/
            ~array() noexcept
            {
                static_assert(MULTI_GPU_ASSERTION(), MULTI_GPU_MSG_NOTE(device::array::<field::SKELETON>, "Need to free all pointers"));
                errorHandler::check(cudaFree(ptr_));
            }

            /**
             * @brief Get read-only access to underlying data
             * @return Const pointer to device memory
             **/
            __device__ __host__ [[nodiscard]] inline const T *constPtr() const noexcept
            {
                static_assert(MULTI_GPU_ASSERTION(), MULTI_GPU_MSG_NOTE(device::array::<field::SKELETON>, "Need to add indexing into pointers for multi GPU"));
                return ptr_;
            }

            /**
             * @brief Get mutable access to underlying data
             * @return Pointer to device memory
             **/
            __device__ __host__ [[nodiscard]] inline T *ptr() noexcept
            {
                static_assert(MULTI_GPU_ASSERTION(), MULTI_GPU_MSG_NOTE(device::array::<field::SKELETON>, "Need to add indexing into pointers for multi GPU"));
                return ptr_;
            }

            /**
             * @brief Provide reference to pointer for swapping operations
             **/
            __host__ [[nodiscard]] inline constexpr T * ptrRestrict & ptrRef() noexcept
            {
                static_assert(MULTI_GPU_ASSERTION(), MULTI_GPU_MSG_NOTE(device::array::<field::SKELETON>, "Need to add indexing into pointers for multi GPU"));
                return ptr_;
            }

        private:
            /**
             * @brief Pointer to the data
             **/
            T *ptrRestrict ptr_;
        };

        template <typename T, class VelocitySet, const time::type TimeType>
        class array<field::FULL_FIELD, T, VelocitySet, TimeType>
        {
        public:
            /**
             * @brief Constructs a device array from host data
             * @tparam MallocType Template parameter for the type of host memory allocation
             * @param[in] hostArray Source data allocated on host memory
             * @param[in] programCtrl The program control object
             * @param[in] allocate Defines whether the object is to be allocated or not
             **/
            template <const host::mallocType MallocType>
            __host__ [[nodiscard]] array(
                const host::array<MallocType, T, VelocitySet, TimeType> &hostArray,
                const programControl &programCtrl,
                const bool allocate = true)
                : ptr_(allocate_on_devices(hostArray.mesh(), hostArray.data(), allocate, programCtrl)),
                  name_(hostArray.name()),
                  mesh_(hostArray.mesh()),
                  programCtrl_(programCtrl)
            {
                initialise_boundary_condition(name_, programCtrl.deviceList());
            };

            /**
             * @brief Constructs a device array from a uniform value
             * @param[in] name Name of the field
             * @param[in] mesh The lattice mesh
             * @param[in] value The uniform value to initialise the array to
             * @param[in] programCtrl The program control object
             * @param[in] allocate Defines whether the object is to be allocated or not
             **/
            __host__ [[nodiscard]] array(
                const name_t &name,
                const host::latticeMesh &mesh,
                const T value,
                const programControl &programCtrl,
                const bool allocate = true)
                : ptr_(allocate_on_devices(mesh, value, allocate, programCtrl)),
                  name_(name),
                  mesh_(mesh),
                  programCtrl_(programCtrl)
            {
                initialise_boundary_condition(name_, programCtrl.deviceList());
            };

            /**
             * @brief Constructs a device array from a checkpoint or initial conditions
             * @param[in] name Name of the field
             * @param[in] mesh The lattice mesh
             * @param[in] programCtrl The program control object
             * @param[in] allocate Defines whether the object is to be allocated or not
             **/
            __host__ [[nodiscard]] array(
                const name_t &name,
                const host::latticeMesh &mesh,
                const programControl &programCtrl,
                const bool allocate = true)
                : ptr_(allocate_on_devices(host::array<host::PAGED, T, VelocitySet, TimeType>(name, mesh, programCtrl), allocate, programCtrl)),
                  name_(name),
                  mesh_(mesh),
                  programCtrl_(programCtrl)
            {
                initialise_boundary_condition(name_, programCtrl.deviceList());
            };

            /**
             * @brief Destructor - automatically releases device memory
             * @note Noexcept guarantee: failsafe if cudaFree fails
             **/
            ~array() noexcept
            {
                const label_t nxGPUs = mesh_.nDevices<axis::X>();
                const label_t nyGPUs = mesh_.nDevices<axis::Y>();
                const label_t nzGPUs = mesh_.nDevices<axis::Z>();

                if constexpr (verbose())
                {
                    std::cout << "Entering device::array destructor for field " << name_ << std::endl;
                }

                if (!(ptr_ == nullptr))
                {
                    gpu_for(
                        nxGPUs, nyGPUs, nzGPUs,
                        [&](const label_t GPU_x, const label_t GPU_y, const label_t GPU_z)
                        {
                            const label_t virtualDeviceIndex = GPU::idx(GPU_x, GPU_y, GPU_z, nxGPUs, nyGPUs);

                            if (!(ptr_[virtualDeviceIndex] == nullptr))
                            {
                                if constexpr (verbose())
                                {
                                    std::cout << "Freeing ptr[" << virtualDeviceIndex << "];" << std::endl;
                                }

                                errorHandler::check(cudaDeviceSynchronize());
                                errorHandler::check(cudaSetDevice(programCtrl_.deviceList()[virtualDeviceIndex]));
                                errorHandler::check(cudaFree(ptr_[virtualDeviceIndex]));
                                errorHandler::check(cudaDeviceSynchronize());

                                if constexpr (verbose())
                                {
                                    std::cout << "Freed ptr[" << virtualDeviceIndex << "];" << std::endl;
                                }
                            }
                        });

                    errorHandler::check(cudaSetDevice(programCtrl_.deviceList()[0]));

                    if constexpr (verbose())
                    {
                        std::cout << "Freeing host pointer collection" << std::endl;
                    }
                    errorHandler::check(cudaFreeHost(ptr_));
                    if constexpr (verbose())
                    {
                        std::cout << "Freed host pointer collection" << std::endl;
                    }
                }
                else
                {
                    if constexpr (verbose())
                    {
                        std::cout << "Nothing to free" << std::endl;
                    }
                }
            }

            /**
             * @brief Get read-only access to underlying data
             * @return Const pointer to device memory
             **/
            template <typename Idx>
            __device__ __host__ [[nodiscard]] inline const T *ptr(const Idx idx) const noexcept
            {
                return ptr_[idx];
            }

            /**
             * @brief Get mutable access to underlying data
             * @return Pointer to device memory
             **/
            template <typename Idx>
            __device__ __host__ [[nodiscard]] inline T *ptr(const Idx idx) noexcept
            {
                return ptr_[idx];
            }

            /**
             * @brief Get array identifier name
             * @return Const reference to name string
             **/
            __host__ [[nodiscard]] inline const name_t &name() const noexcept
            {
                return name_;
            }

            /**
             * @brief Get associated mesh object
             * @return Const reference to lattice mesh
             **/
            __host__ [[nodiscard]] inline const host::latticeMesh &mesh() const noexcept
            {
                return mesh_;
            }

            /**
             * @brief Get total number of elements
             * @return Number of elements in array
             * @note Returns mesh point count - assumes 1:1 element-to-point mapping
             **/
            template <typename SizeType = label_t>
            __host__ [[nodiscard]] inline constexpr SizeType size() const noexcept
            {
                return mesh_.nPoints<SizeType>();
            }

            /**
             * @brief Returns the time type of the array
             **/
            __host__ [[nodiscard]] inline consteval time::type timeType() const noexcept
            {
                return TimeType;
            }

            /**
             * @brief Copies the array to a host pointer
             * @param hostPtr Pointer to memory allocated on the host
             **/
            __host__ void copy_to_host(T *const ptrRestrict hostPtr)
            {
                static_assert(MULTI_GPU_ASSERTION());

                constexpr const std::size_t N = 1;

                const std::size_t nDevices = mesh_.nDevices<axis::X>() * mesh_.nDevices<axis::Y>() * mesh_.nDevices<axis::Z>();

                const std::size_t nxPointsPerGPU = mesh_.nx<std::size_t>() / mesh_.nDevices<axis::X, std::size_t>();
                const std::size_t nyPointsPerGPU = mesh_.ny<std::size_t>() / mesh_.nDevices<axis::Y, std::size_t>();
                const std::size_t nzPointsPerGPU = mesh_.nz<std::size_t>() / mesh_.nDevices<axis::Z, std::size_t>();
                const std::size_t nPointsPerGPU = nxPointsPerGPU * nyPointsPerGPU * nzPointsPerGPU;

                for (std::size_t virtualDeviceIndex = 0; virtualDeviceIndex < nDevices; virtualDeviceIndex++)
                {
                    const label_t startIndex = virtualDeviceIndex * nPointsPerGPU;

                    errorHandler::check(cudaMemcpy(&(hostPtr[startIndex]), ptr_[virtualDeviceIndex], nPointsPerGPU * sizeof(T), cudaMemcpyDeviceToHost));
                }
            }

        private:
            /**
             * @brief The underlying pointers to device memory
             **/
            T **const ptrRestrict ptr_;

            /**
             * @brief Names of the solution variables
             **/
            const name_t name_;

            /**
             * @brief Reference to the mesh
             **/
            const host::latticeMesh &mesh_;

            /**
             * @brief Reference to the program control
             **/
            const programControl &programCtrl_;

            /**
             * @brief Allocates all partitions of the array to the devices
             * @param[in] mesh The mesh
             * @param[in] hostArrayGlobal Pointer to the array allocated on the host
             **/
            __host__ [[nodiscard]] static T **allocate_on_devices(
                const host::latticeMesh &mesh,
                const T *hostArrayGlobal,
                const bool allocate,
                const programControl &programCtrl)
            {
                if (allocate)
                {
                    const label_t nxGPUs = mesh.nDevices<axis::X>();
                    const label_t nyGPUs = mesh.nDevices<axis::Y>();
                    const label_t nzGPUs = mesh.nDevices<axis::Z>();

                    const std::size_t nDevices = mesh.nDevices<axis::X, std::size_t>() * mesh.nDevices<axis::Y, std::size_t>() * mesh.nDevices<axis::Z, std::size_t>();

                    T **hostPtrsToDevice = host::allocate<T *>(nDevices, nullptr);

                    gpu_for(
                        nxGPUs, nyGPUs, nzGPUs,
                        [&](const label_t GPU_x, const label_t GPU_y, const label_t GPU_z)
                        {
                            const label_t virtualDeviceIndex = GPU::idx(GPU_x, GPU_y, GPU_z, nxGPUs, nyGPUs);

                            hostPtrsToDevice[virtualDeviceIndex] = allocate_device_segment(mesh, hostArrayGlobal, GPU_x, GPU_y, GPU_z, programCtrl);
                        });

                    return hostPtrsToDevice;
                }
                else
                {
                    return nullptr;
                }
            }

            /**
             * @brief Partitions and allocates an existing std::vector on the devices
             * @param[in] hostArrayGlobal Pointer to the array allocated on the host
             **/
            __host__ [[nodiscard]] static T **allocate_on_devices(
                const host::latticeMesh &mesh,
                const std::vector<T> &hostArrayGlobal,
                const bool allocate,
                const programControl &programCtrl)
            {
                return allocate_on_devices(mesh, hostArrayGlobal.data(), allocate, programCtrl);
            }

            /**
             * @brief Partitions and allocates an existing host::array on the devices
             * @param[in] hostArrayGlobal Pointer to the array allocated on the host
             **/
            template <const host::mallocType MallocType>
            __host__ [[nodiscard]] T **allocate_on_devices(
                const host::array<MallocType, T, VelocitySet, TimeType> &hostArrayGlobal,
                const bool allocate,
                const programControl &programCtrl)
            {
                return allocate_on_devices(hostArrayGlobal.mesh(), hostArrayGlobal.data(), allocate, programCtrl);
            }

            /**
             * @brief Allocates a uniform value distributed amongst the devices
             * @param[in] mesh The mesh
             * @param[in] hostArrayGlobal Pointer to the array allocated on the host
             **/
            __host__ [[nodiscard]] T **allocate_on_devices(
                const host::latticeMesh &mesh,
                const T val,
                const bool allocate,
                const programControl &programCtrl)
            {
                const std::vector<T> toAllocate(static_cast<std::size_t>(allocate) * mesh.nPoints<std::size_t>(), val);
                return allocate_on_devices(mesh, toAllocate, allocate, programCtrl);
            }

            /**
             * @brief Creates a partition of the mesh and allocates it to a pointer
             * @param[in] mesh The mesh
             * @param[in] hostArrayGlobal Pointer to the array allocated on the host
             * @param[in] GPU_x, GPU_y, GPUz Indices of the device to allocate on
             **/
            __host__ [[nodiscard]] static T *allocate_device_segment(
                const host::latticeMesh &mesh,
                const T *hostArrayGlobal,
                const label_t GPU_x,
                const label_t GPU_y,
                const label_t GPU_z,
                const programControl &programCtrl)
            {
                const label_t nxGPUs = mesh.nDevices<axis::X>();
                const label_t nyGPUs = mesh.nDevices<axis::Y>();
                const label_t nzGPUs = mesh.nDevices<axis::Z>();
                const label_t nxPointsPerGPU = mesh.nx() / nxGPUs;
                const label_t nyPointsPerGPU = mesh.ny() / nyGPUs;
                const label_t nzPointsPerGPU = mesh.nz() / nzGPUs;
                const label_t nPointsPerGPU = nxPointsPerGPU * nyPointsPerGPU * nzPointsPerGPU;
                const label_t virtualDeviceIndex = GPU::idx(GPU_x, GPU_y, GPU_z, nxGPUs, nyGPUs);
                const label_t startIndex = virtualDeviceIndex * nPointsPerGPU;

                T *devPtr = device::allocate<T>(nPointsPerGPU, programCtrl.deviceList()[virtualDeviceIndex]);

                device::copy(devPtr, &(hostArrayGlobal[startIndex]), nPointsPerGPU, programCtrl.deviceList()[virtualDeviceIndex]);

                return devPtr;
            }

            /**
             * @brief Copies the underlying std::vector of a host::array type to the device
             * @param[in] hostArray The host::array to be copied to the device
             * @return A pointer to the copied data
             **/
            template <const host::mallocType MallocType>
            __host__ [[nodiscard]] T *to_device(const host::array<MallocType, T, VelocitySet, TimeType> &hostArray)
            {
                return device::allocateArray<T>(hostArray.arr());
            }

            /**
             * @brief Converts a variable name to an index
             * @param[in] name The name of the variable
             * @return The index of the variable, or -1 if not found
             **/
            __host__ [[nodiscard]] static inline label_t name_to_index(const name_t &name) noexcept
            {
                if (name == "u")
                {
                    return 0;
                }
                else if (name == "v")
                {
                    return 1;
                }
                else if (name == "w")
                {
                    return 2;
                }
                return static_cast<label_t>(-1);
            }

            /**
             * @brief Initialises boundary condition values on the GPU for a given variable name
             * @param[in] name The name of the variable to initialise boundary conditions for
             * @param[in] devicelist List of devices to allocate the constants over
             **/
            __host__ static void initialise_boundary_condition(const name_t &name, const std::vector<deviceIndex_t> &deviceList) noexcept
            {
                static_assert(MULTI_GPU_ASSERTION(), MULTI_GPU_MSG_NOTE(device::array::initialise_boundary_condition, "Believed to be correct"));

                if ((name == "u") || (name == "v") || (name == "w"))
                {
                    const label_t i = name_to_index(name);

                    const boundaryValue<VelocitySet, false> North(name, "North");
                    const boundaryValue<VelocitySet, false> South(name, "South");
                    const boundaryValue<VelocitySet, false> East(name, "East");
                    const boundaryValue<VelocitySet, false> West(name, "West");
                    const boundaryValue<VelocitySet, false> Back(name, "Back");
                    const boundaryValue<VelocitySet, false> Front(name, "Front");

                    for (std::size_t virtualDeviceIndex = 0; virtualDeviceIndex < deviceList.size(); virtualDeviceIndex++)
                    {
                        errorHandler::check(cudaSetDevice(deviceList[virtualDeviceIndex]));
                        device::copyToSymbol(device::U_North, North(), i);
                        device::copyToSymbol(device::U_South, South(), i);
                        device::copyToSymbol(device::U_East, East(), i);
                        device::copyToSymbol(device::U_West, West(), i);
                        device::copyToSymbol(device::U_Back, Back(), i);
                        device::copyToSymbol(device::U_Front, Front(), i);
                    }
                }
            }
        };
    }
}

#endif

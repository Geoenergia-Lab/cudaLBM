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
    Memory management routines for the LBM code

Namespace
    LBM::host, LBM::device

SourceFiles
    memory.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_MEMORY_CUH
#define __MBLBM_MEMORY_CUH

#include "../LBMIncludes.cuh"
#include "../typedefs/typedefs.cuh"
#include "../globalFunctions.cuh"

namespace LBM
{
    template <typename T>
    __host__ void allocateMessage(const name_t &functionName, const host::label_t nPoints, const T *ptr) noexcept
    {
        std::cout << "Allocated " << sizeof(T) * nPoints << " bytes of memory in " << functionName << " to address " << ptr << " (current device: " << GPU::current_ordinal() << ")" << std::endl;
    }

    template <typename T>
    __host__ void copyMessage(const name_t &functionName, const host::label_t nPoints, const T *srcPtr, const T *destPtr) noexcept
    {
        std::cout << "Copied " << sizeof(T) * nPoints << " bytes of memory in " << functionName << " from address " << srcPtr << " to address " << destPtr << " (current device: " << GPU::current_ordinal() << ")" << std::endl;
    }

    namespace host
    {
        /**
         * @brief Allocates pinned memory on the host
         * @tparam T The type of memory to be allocated
         * @param[in] ptr The pointer to be allocated on the host
         * @param[in] nPoints The number of points of type T to be allocated
         **/
        template <typename T>
        __host__ void allocateMemory(T **ptr, const host::label_t nPoints) noexcept
        {
            errorHandler::check(cudaMallocHost(ptr, sizeof(T) * nPoints));
        }

        /**
         * @brief Allocates pinned memory on the host, initialises it to val and returns a pointer
         * @tparam T The type of memory to be allocated
         * @param[in] nPoints The number of points of type T to be allocated
         * @param[in] val The value to initialise all elements of the block of memory
         * @return A pointer to a block of pinned memory on the host, all initialised to val
         **/
        template <typename T>
        __host__ [[nodiscard]] T *allocate(const host::label_t nPoints, const T val) noexcept
        {
            T *ptr;

            allocateMemory(&ptr, nPoints);

            // if constexpr (verbose())
            // {
            //     allocateMessage("host::allocate", nPoints, ptr);
            // }

            std::uninitialized_fill_n(ptr, nPoints, val);

            return ptr;
        }

        /**
         * @brief Copies from a pointer on the device to a pointer on the GPU
         * @tparam T The type of memory to be copied
         * @param[in] devPtr The pointer on the device to be copied from
         * @param[in] hostPtr The pointer on the host to be copied to
         * @param[in] fieldIndex The index of the field in the host buffer to copy to
         * @param[in] nPoints The number of points to be copied to the host
         * @throws std::runtime_error if CUDA memory copy fails or if the pointer is null
         **/
        template <typename T>
        __host__ void to_host(const T *const ptrRestrict devPtr, T *const ptrRestrict hostPtr, const device::label_t fieldIndex, const host::label_t nPoints) noexcept
        {
            errorHandler::check(cudaMemcpy(hostPtr + (fieldIndex * nPoints), devPtr, nPoints * sizeof(T), cudaMemcpyDeviceToHost));

            if constexpr (verbose())
            {
                copyMessage("host::to_host", nPoints, devPtr, hostPtr);
            }
        }

        /**
         * @brief Copies data from device memory to host memory
         * @tparam T Data type of the elements
         * @param[in] devPtr Pointer to device memory to copy from
         * @param[in] nPoints Number of elements to copy
         * @return std::vector<T> containing the copied data
         * @throws std::runtime_error if CUDA memory copy fails
         **/
        template <typename T>
        __host__ [[nodiscard]] const std::vector<T> to_host(const T *const ptrRestrict devPtr, const host::label_t nPoints) noexcept
        {
            std::vector<T> hostFields(nPoints, 0);

            to_host(devPtr, hostFields, 0, nPoints);

            return hostFields;
        }
    }

    namespace device
    {
        /**
         * @brief Allocates memory on the device
         * @tparam T Data type to allocate
         * @param[out] ptr Pointer to be allocated
         * @param[in] nPoints Number of elements to allocate
         * @throws std::runtime_error if CUDA allocation fails
         **/
        template <typename T>
        __host__ void allocateMemory(T **ptr, const host::label_t nPoints) noexcept
        {
            errorHandler::check(cudaDeviceSynchronize());

            errorHandler::check(cudaMalloc(ptr, sizeof(T) * nPoints));

            errorHandler::check(cudaDeviceSynchronize());
        }

        /**
         * @brief Allocates and returns a pointer to device memory
         * @tparam T Data type to allocate
         * @param[in] nPoints Number of elements to allocate
         * @return Pointer to allocated device memory
         * @throws std::runtime_error if CUDA allocation fails
         * @note Verbose mode prints allocation details
         **/
        template <typename T>
        __host__ [[nodiscard]] T *allocate(const host::label_t nPoints) noexcept
        {
            T *ptr;

            errorHandler::check(cudaDeviceSynchronize());

            allocateMemory(&ptr, nPoints);

            errorHandler::check(cudaDeviceSynchronize());

            if constexpr (verbose())
            {
                allocateMessage("device::allocate", nPoints, ptr);
            }

            return ptr;
        }

        /**
         * @overload Allocates memory on a specific device
         * @param[in] nPoints Number of elements to allocate
         * @param[in] deviceID The device on which to allocate the memory
         **/
        template <typename T>
        __host__ [[nodiscard]] T *allocate(const host::label_t nPoints, const deviceIndex_t deviceID) noexcept
        {
            errorHandler::check(cudaDeviceSynchronize());

            errorHandler::check(cudaSetDevice(deviceID));

            errorHandler::check(cudaDeviceSynchronize());

            return allocate<T>(nPoints);
        }

        template <typename T>
        __host__ void free(const T *ptr, const deviceIndex_t deviceID)
        {
            errorHandler::check(cudaDeviceSynchronize());

            errorHandler::check(cudaSetDevice(deviceID));

            errorHandler::check(cudaDeviceSynchronize());

            errorHandler::check(cudaFree(const_cast<T *>(ptr)));
        }

        template <typename T>
        __host__ void free(T *ptr, const deviceIndex_t deviceID)
        {
            errorHandler::check(cudaDeviceSynchronize());

            errorHandler::check(cudaSetDevice(deviceID));

            errorHandler::check(cudaDeviceSynchronize());

            errorHandler::check(cudaFree(ptr));
        }

        /**
         * @brief Copies data from host to device memory
         * @tparam T Data type of the elements
         * @param[out] devPtr Destination device pointer
         * @param[in] hostPtr Source host pointer
         * @param[in] nPoints The number of points of T to copy to the device
         * @throws std::runtime_error if CUDA memory copy fails
         * @note Verbose mode prints copy details
         **/
        template <typename T>
        __host__ void copy(T *const devPtr, const T *const ptrRestrict hostPtr, const host::label_t nPoints) noexcept
        {
            errorHandler::check(cudaDeviceSynchronize());

            errorHandler::check(cudaMemcpy(devPtr, hostPtr, nPoints * sizeof(T), cudaMemcpyHostToDevice));

            errorHandler::check(cudaDeviceSynchronize());

            if constexpr (verbose())
            {
                copyMessage("device::copy", nPoints, hostPtr, devPtr);
            }
        }

        template <typename T>
        __host__ void copy(T *const devPtr, const T *const ptrRestrict hostPtr, const host::label_t nPoints, const deviceIndex_t deviceID) noexcept
        {
            errorHandler::check(cudaDeviceSynchronize());

            errorHandler::check(cudaSetDevice(deviceID));

            errorHandler::check(cudaDeviceSynchronize());

            copy(devPtr, hostPtr, nPoints);

            errorHandler::check(cudaDeviceSynchronize());
        }

        /**
         * @brief Copies data from host to device memory
         * @tparam T Data type of the elements
         * @param[out] ptr Destination device pointer
         * @param[in] f Source host vector
         * @throws std::runtime_error if CUDA memory copy fails
         * @note Verbose mode prints copy details
         **/
        template <typename T>
        __host__ void copy(T *const ptr, const std::vector<T> &f) noexcept
        {
            copy(ptr, f.data(), f.size());
        }

        /**
         * @overload Copies to a specific device
         * @param[out] ptr Destination device pointer
         * @param[in] f Source host vector
         * @param[in] deviceID The device on which to allocate the memory
         **/
        template <typename T>
        __host__ void copy(T *const ptr, const std::vector<T> &f, const deviceIndex_t deviceID) noexcept
        {
            copy(ptr, f.data(), f.size(), deviceID);
        }

        /**
         * @brief Allocates device memory and copies host data to it
         * @tparam T Data type of the elements
         * @param[in] f Host vector to copy to device
         * @return Pointer to allocated device memory containing copied data
         * @throws std::runtime_error if CUDA operations fail
         **/
        template <typename T>
        __host__ [[nodiscard]] T *allocateArray(const std::vector<T> &f) noexcept
        {
            errorHandler::check(cudaDeviceSynchronize());

            T *ptr = allocate<T>(f.size());

            errorHandler::check(cudaDeviceSynchronize());

            copy(ptr, f);

            errorHandler::check(cudaDeviceSynchronize());

            return ptr;
        }

        /**
         * @overload Allocates and copies to memory on a specific device
         * @param[in] f Host vector to copy to device
         * @param[in] deviceID The device on which to allocate the memory
         **/
        template <typename T>
        __host__ [[nodiscard]] T *allocateArray(const std::vector<T> &f, const deviceIndex_t deviceID) noexcept
        {
            errorHandler::check(cudaSetDevice(deviceID));

            return allocateArray(f);
        }

        /**
         * @brief Allocates device memory and initializes it with a value
         * @tparam T Data type of the elements
         * @param[in] nPoints Number of elements to allocate
         * @param[in] val Value to initialize all elements with
         * @return Pointer to allocated and initialized device memory
         * @throws std::runtime_error if CUDA operations fail
         **/
        template <typename T>
        __host__ [[nodiscard]] T *allocateArray(const device::label_t nPoints, const T val) noexcept
        {
            errorHandler::check(cudaDeviceSynchronize());

            T *ptr = allocate<T>(nPoints);

            errorHandler::check(cudaDeviceSynchronize());

            copy(ptr, std::vector<T>(nPoints, val));

            errorHandler::check(cudaDeviceSynchronize());

            return ptr;
        }

        /**
         * @brief Allocates device memory and initializes it with a value on a specific device
         * @param[in] nPoints Number of elements to allocate
         * @param[in] val Value to initialize all elements with
         * @param[in] deviceID The device on which to allocate the memory
         **/
        template <typename T>
        __host__ [[nodiscard]] T *allocateArray(const device::label_t nPoints, const T val, const deviceIndex_t deviceID) noexcept
        {
            errorHandler::check(cudaDeviceSynchronize());

            errorHandler::check(cudaSetDevice(deviceID));

            errorHandler::check(cudaDeviceSynchronize());

            return allocateArray(nPoints, val);
        }
    }
}

#include "cache.cuh"

#endif
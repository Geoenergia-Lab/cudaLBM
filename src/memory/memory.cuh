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
    /**
     * @brief Executes the given callable only if the allocation status is GOOD.
     * @tparam F Callable type.
     * @tparam Args Argument types.
     * @param[in] f The callable to execute conditionally.
     * @param[in] args Arguments to forward to the callable.
     **/
    template <typename F, typename... Args>
    __host__ void ifAllocationAllowed(F &&f, Args &&...args) noexcept
    {
        if (runTime::program_status.load() == runTime::GOOD)
        {
            std::forward<F>(f)(std::forward<Args>(args)...);
        }
    }

    /**
     * @brief Prints an allocation message if verbose output is enabled.
     * @tparam T Data type of the allocated memory.
     * @param[in] functionName Name of the calling function.
     * @param[in] nPoints Number of elements allocated.
     * @param[in] ptr Pointer to the allocated memory.
     **/
    template <typename T>
    __host__ void allocateMessage(const name_t &functionName, const host::label_t nPoints, const T *ptr) noexcept
    {
        std::cout << "Allocated " << sizeof(T) * nPoints << " bytes of memory in " << functionName << " to address " << ptr << " (current device: " << device::current_ordinal() << ")" << std::endl;
    }

    /**
     * @brief Prints a copy message if verbose output is enabled.
     * @tparam T Data type of the copied memory.
     * @param[in] functionName Name of the calling function.
     * @param[in] nPoints Number of elements copied.
     * @param[in] srcPtr Source pointer.
     * @param[in] destPtr Destination pointer.
     **/
    template <typename T>
    __host__ void copyMessage(const name_t &functionName, const host::label_t nPoints, const T *srcPtr, const T *destPtr) noexcept
    {
        std::cout << "Copied " << sizeof(T) * nPoints << " bytes of memory in " << functionName << " from address " << srcPtr << " to address " << destPtr << " (current device: " << device::current_ordinal() << ")" << std::endl;
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
            *ptr = nullptr;
            ifAllocationAllowed(
                [&]()
                {
                    errorHandler::handle(cudaMallocHost(ptr, sizeof(T) * nPoints));
                });
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

            if constexpr (verbose())
            {
                allocateMessage("host::allocate", nPoints, ptr);
            }

            ifAllocationAllowed(
                [&]()
                {
                    std::uninitialized_fill_n(ptr, nPoints, val);
                });

            return ptr;
        }

        /**
         * @brief Frees pinned host memory previously allocated with host::allocateMemory.
         * @tparam T Data type of the memory.
         * @param[in] ptr Pointer to the memory to free. May be nullptr.
         **/
        template <typename T>
        __host__ void free(T *const ptrRestrict ptr) noexcept
        {
            if (ptr != nullptr)
            {
                errorHandler::handle(cudaFreeHost(ptr));
            }
        }

        /**
         * @brief Frees pinned host memory (const overload).
         * @tparam T Data type of the memory.
         * @param[in] ptr Pointer to the memory to free. May be nullptr.
         **/
        template <typename T>
        __host__ void free(const T *const ptrRestrict ptr) noexcept
        {
            free(const_cast<T *>(ptr));
        }
    }

    namespace device
    {
        /**
         * @brief Sets the current CUDA device.
         * @param[in] deviceID The device index to set as current.
         **/
        __host__ void setDevice(const deviceIndex_t deviceID) noexcept
        {
            ifAllocationAllowed(
                [&]()
                {
                    errorHandler::handle(cudaSetDevice(deviceID));
                });
        }

        /**
         * @brief Synchronizes the current CUDA device.
         **/
        __host__ void syncDevice() noexcept
        {
            ifAllocationAllowed(
                [&]()
                {
                    errorHandler::handle(cudaDeviceSynchronize());
                });
        }

        /**
         * @brief Allocates a symbol of type T to the device
         * @tparam T The type of the pointer to copy to
         * @param[in] symbol The symbol to which the value is to be copied
         * @param[in] value The value to copy to the symbol
         **/
        template <typename T>
        __host__ void copyToSymbol(const T &symbol, const T value) noexcept
        {
            syncDevice();
            const T valueTemp = value;

            ifAllocationAllowed(
                [&]()
                {
                    errorHandler::handle(cudaMemcpyToSymbol(symbol, &valueTemp, sizeof(T), 0, cudaMemcpyHostToDevice));
                });

            syncDevice();
        }

        /**
         * @brief Allocates an array of type T and size N to the device
         * @tparam T The type of the pointer to copy to
         * @tparam N Number of elements of the symbol on the device
         * @param[in] symbol The symbol to which the value is to be copied
         * @param[in] value The value to copy to the symbol
         **/
        template <typename T, const host::label_t N>
        __host__ void copyToSymbol(const T (&symbol)[N], const T (&value)[N]) noexcept
        {
            syncDevice();

            ifAllocationAllowed(
                [&]()
                {
                    errorHandler::handle(cudaMemcpyToSymbol(symbol, value, N * sizeof(T), 0, cudaMemcpyHostToDevice));
                });

            syncDevice();
        }

        /**
         * @brief Allocates a symbol of type T to an array on the device
         * @tparam T The type of the pointer to copy to
         * @tparam N Number of elements of the symbol on the device
         * @tparam SizeType Index type
         * @param[in] symbol The array to which the value is to be copied
         * @param[in] value The value to copy to the symbol
         * @param[in] index The index in the array to copy the value to
         **/
        template <typename T, const host::label_t N, typename SizeType>
        __host__ void copyToSymbol(const T (&symbol)[N], const T value, const SizeType index) noexcept
        {
            if (static_cast<host::label_t>(index) >= N)
            {
                errorHandler::handle(cudaErrorMemoryAllocation);
            }
            syncDevice();
            const T valueTemp = value;

            ifAllocationAllowed(
                [&]()
                {
                    errorHandler::handle(cudaMemcpyToSymbol(symbol, &valueTemp, static_cast<host::label_t>(sizeof(T)), static_cast<host::label_t>(index) * static_cast<host::label_t>(sizeof(T)), cudaMemcpyHostToDevice));
                });

            syncDevice();
        }

        /**
         * @brief Allocates memory on the device
         * @tparam T Data type to allocate
         * @param[out] ptr Pointer to be allocated
         * @param[in] nPoints Number of elements to allocate
         **/
        template <typename T>
        __host__ void allocateMemory(T **ptr, const host::label_t nPoints) noexcept
        {
            syncDevice();

            *ptr = nullptr;

            const host::label_t nBytes = sizeof(T) * nPoints;

            ifAllocationAllowed(
                [&]()
                {
                    host::label_t free_bytes = 0;
                    host::label_t total_bytes = 0;
                    errorHandler::handle(cudaMemGetInfo(&free_bytes, &total_bytes));
                    if ((nBytes < free_bytes) && (nBytes < total_bytes))
                    {
                        errorHandler::handle(cudaMalloc(ptr, nBytes));
                    }
                    else
                    {
                        errorHandler::handle(cudaErrorMemoryAllocation);
                    }
                });

            syncDevice();
        }

        /**
         * @brief Allocates and returns a pointer to device memory
         * @tparam T Data type to allocate
         * @param[in] nPoints Number of elements to allocate
         * @return Pointer to allocated device memory
         * @note Verbose mode prints allocation details
         **/
        template <typename T>
        __host__ [[nodiscard]] T *allocate(const host::label_t nPoints) noexcept
        {
            T *ptr;

            allocateMemory(&ptr, nPoints);

            if constexpr (verbose())
            {
                allocateMessage("device::allocate", nPoints, ptr);
            }

            return ptr;
        }

        /**
         * @overload Allocates memory on a specific device
         * @tparam T The type of the returned pointer
         * @param[in] nPoints Number of elements to allocate
         * @param[in] deviceID The device on which to allocate the memory
         **/
        template <typename T>
        __host__ [[nodiscard]] T *allocate(const host::label_t nPoints, const deviceIndex_t deviceID) noexcept
        {
            syncDevice();

            setDevice(deviceID);

            syncDevice();

            return allocate<T>(nPoints);
        }

        /**
         * @brief Frees device memory allocated with device::allocate.
         * @tparam T Data type of the memory.
         * @param[in] ptr Pointer to the device memory to free. May be nullptr.
         **/
        template <typename T>
        __host__ void free(T *const ptrRestrict ptr) noexcept
        {
            cudaPointerAttributes attrs;
            errorHandler::handle(cudaPointerGetAttributes(&attrs, ptr));
            if (ptr == nullptr)
            {
                return;
            }
            else if (attrs.type == cudaMemoryTypeDevice)
            {
                syncDevice();
                setDevice(attrs.device);
                syncDevice();
                errorHandler::handle(cudaFree(ptr));
                syncDevice();
            }
        }

        /**
         * @brief Frees device memory (const overload).
         * @tparam T Data type of the memory.
         * @param[in] ptr Pointer to the device memory to free. May be nullptr.
         **/
        template <typename T>
        __host__ void free(const T *const ptrRestrict ptr) noexcept
        {
            free(const_cast<T *>(ptr));
        }

        /**
         * @brief Copies data from host to device memory
         * @tparam T Data type of the elements
         * @param[out] devPtr Destination device pointer
         * @param[in] hostPtr Source host pointer
         * @param[in] nPoints The number of points of T to copy to the device
         * @note Verbose mode prints copy details
         **/
        template <typename T>
        __host__ void copy(T *const devPtr, const T *const ptrRestrict hostPtr, const host::label_t nPoints) noexcept
        {
            syncDevice();

            if (devPtr == nullptr)
            {
                errorHandler::handle(cudaErrorMemoryAllocation);
            }

            if (hostPtr == nullptr)
            {
                errorHandler::handle(cudaErrorMemoryAllocation);
            }

            ifAllocationAllowed(
                [&]()
                {
                    errorHandler::handle(cudaMemcpy(devPtr, hostPtr, nPoints * sizeof(T), cudaMemcpyHostToDevice));
                });

            syncDevice();

            if constexpr (verbose())
            {
                copyMessage("device::copy", nPoints, hostPtr, devPtr);
            }
        }

        /**
         * @brief Copies data from host to device memory on a specified device.
         * @tparam T Data type of the elements.
         * @param[out] devPtr Destination device pointer.
         * @param[in] hostPtr Source host pointer.
         * @param[in] nPoints Number of elements to copy.
         * @param[in] deviceID The device on which to perform the copy.
         **/
        template <typename T>
        __host__ void copy(T *const devPtr, const T *const ptrRestrict hostPtr, const host::label_t nPoints, const deviceIndex_t deviceID) noexcept
        {
            syncDevice();

            setDevice(deviceID);

            syncDevice();

            copy(devPtr, hostPtr, nPoints);

            syncDevice();
        }

        /**
         * @brief Copies data from host to device memory
         * @tparam T Data type of the elements
         * @param[out] ptr Destination device pointer
         * @param[in] f Source host vector
         * @note Verbose mode prints copy details
         **/
        template <typename T>
        __host__ void copy(T *const ptr, const std::vector<T> &f) noexcept
        {
            copy(ptr, f.data(), f.size());
        }

        /**
         * @overload Copies to a specific device
         * @tparam T The type of the pointer to copy to
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
         **/
        template <typename T>
        __host__ [[nodiscard]] T *allocateArray(const std::vector<T> &f) noexcept
        {
            syncDevice();

            T *ptr = allocate<T>(f.size());

            syncDevice();

            copy(ptr, f);

            syncDevice();

            return ptr;
        }

        /**
         * @overload Allocates and copies to memory on a specific device
         * @tparam T The type of the returned pointer
         * @param[in] f Host vector to copy to device
         * @param[in] deviceID The device on which to allocate the memory
         **/
        template <typename T>
        __host__ [[nodiscard]] T *allocateArray(const std::vector<T> &f, const deviceIndex_t deviceID) noexcept
        {
            setDevice(deviceID);

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
            syncDevice();

            T *ptr = allocate<T>(nPoints);

            syncDevice();

            copy(ptr, std::vector<T>(nPoints, val));

            syncDevice();

            return ptr;
        }

        /**
         * @brief Allocates device memory and initializes it with a value on a specific device
         * @tparam T The type of the returned pointer
         * @param[in] nPoints Number of elements to allocate
         * @param[in] val Value to initialize all elements with
         * @param[in] deviceID The device on which to allocate the memory
         **/
        template <typename T>
        __host__ [[nodiscard]] T *allocateArray(const device::label_t nPoints, const T val, const deviceIndex_t deviceID) noexcept
        {
            syncDevice();

            setDevice(deviceID);

            syncDevice();

            return allocateArray(nPoints, val);
        }

        /**
         * @brief Wrapper for calls to cudaMemcpyAsync copying device to host
         * @tparam T The type of the pointer
         * @param[in] hostPtr Pointer on the host to copy to
         * @param[in] devPtr Pointer on the device to copy from
         * @param[in] nPoints Number of points of the size of T to copy
         * @param[in] stream Stream on which to execute the copy
         **/
        template <typename T>
        __host__ inline void memcpyAsyncDeviceToHost(T *const ptrRestrict hostPtr, const T *const ptrRestrict devPtr, const host::label_t nPoints, const cudaStream_t &stream) noexcept
        {
            errorHandler::handle(cudaMemcpyAsync(hostPtr, devPtr, nPoints * sizeof(T), cudaMemcpyDeviceToHost, stream));
        }

        /**
         * @brief Wrapper for calls to cudaMemcpyAsync copying device to host
         * @tparam T The type of the pointer
         * @param[in] destPtr Pointer on the device to copy to
         * @param[in] destDevice Device to copy to
         * @param[in] srcPtr Pointer on the device to copy from
         * @param[in] srcDevice Device to copy from
         * @param[in] nPoints Number of points to copy
         * @param[in] stream Device execution stream to copy over
         **/
        template <typename T>
        __host__ inline void memcpyPeerAsync(T *const ptrRestrict destPtr, const deviceIndex_t destDevice, const T *const ptrRestrict srcPtr, const deviceIndex_t srcDevice, const host::label_t nPoints, const cudaStream_t &stream) noexcept
        {
            errorHandler::handleInline(cudaMemcpyPeerAsync(destPtr, destDevice, srcPtr, srcDevice, nPoints * sizeof(T), stream));
        }
    }
}

#include "cache.cuh"

#endif
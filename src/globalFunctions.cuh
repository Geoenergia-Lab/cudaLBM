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
    Global utility functions and structures for LBM simulations.
    Contains core functionalities used throughout the LBM implementation.

Namespace
    LBM

SourceFiles
    globalFunctions.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_GLOBALFUNCTIONS_CUH
#define __MBLBM_GLOBALFUNCTIONS_CUH

#include "LBMIncludes.cuh"
#include "typedefs/typedefs.cuh"
#include "runTime/runTime.cuh"

namespace LBM
{
    namespace global
    {
        /**
         * @brief Memory index (generic version)
         * @tparam T The return type
         * @tparam nx Block dimensions in the X axis
         * @tparam ny Block dimensions in the Y axis
         * @tparam nz Block dimensions in the Z axis
         * @param[in] tx Thread x-coordinate within block
         * @param[in] ty Thread y-coordinate within block
         * @param[in] tz Thread z-coordinate within block
         * @param[in] bx Block index in the x-direction
         * @param[in] by Block index in the y-direction
         * @param[in] bz Block index in the z-direction
         * @param[in] nxBlocks Number of blocks in x-direction
         * @param[in] nyBlocks Number of blocks in y-direction
         * @return Linearized index using mesh constants
         *
         * Layout: [bx][by][bz][tz][ty][tx] (tx fastest varying)
         **/
        template <typename T, const T nx, const T ny, const T nz>
        __device__ __host__ [[nodiscard]] inline constexpr T idx(const T tx, const T ty, const T tz, const T bx, const T by, const T bz, const T nxBlocks, const T nyBlocks) noexcept
        {
            return (tx + nx * (ty + ny * (tz + nz * (bx + nxBlocks * (by + nyBlocks * bz)))));
        }
    }

    namespace host
    {
        /**
         * @brief Compile-time recursive loop unroller
         * @tparam Start Starting index (inclusive)
         * @tparam End Ending index (exclusive)
         * @tparam F Callable type accepting integral_constant<host::label_t>
         * @param[in] f Function object to execute per iteration
         *
         * @note Equivalent to runtime loop: `for(host::label_t i=Start; i<End; ++i)`
         * @note Enables `if constexpr` usage in loop bodies
         * @warning Recursion depth limited by compiler constraints
         *
         * Example usage:
         * @code
         * constexpr_for<0, 5>([](auto i) {
         *     // i is integral_constant<host::label_t, N>
         *     if constexpr (i.value % 2 == 0) { ... }
         * });
         * @endcode
         **/
        template <const host::label_t Start, const host::label_t End, typename F>
        __host__ inline constexpr void constexpr_for(F &&f) noexcept
        {
            if constexpr (Start < End)
            {
                f(std::integral_constant<host::label_t, Start>());
                if constexpr (Start + 1 < End)
                {
                    host::constexpr_for<Start + 1, End>(std::forward<F>(f));
                }
            }
        }
    }

    namespace device
    {
        /**
         * @brief Compile-time recursive loop unroller
         * @tparam Start Starting index (inclusive)
         * @tparam End Ending index (exclusive)
         * @tparam F Callable type accepting integral_constant<device::label_t>
         * @param[in] f Function object to execute per iteration
         *
         * @note Equivalent to runtime loop: `for(device::label_t i=Start; i<End; ++i)`
         * @note Enables `if constexpr` usage in loop bodies
         * @warning Recursion depth limited by compiler constraints
         *
         * Example usage:
         * @code
         * constexpr_for<0, 5>([](auto i) {
         *     // i is integral_constant<device::label_t, N>
         *     if constexpr (i.value % 2 == 0) { ... }
         * });
         * @endcode
         **/
        template <const device::label_t Start, const device::label_t End, typename F>
        __device__ inline constexpr void constexpr_for(F &&f) noexcept
        {
            if constexpr (Start < End)
            {
                f(integralConstant<device::label_t, Start>());
                if constexpr (Start + 1 < End)
                {
                    device::constexpr_for<Start + 1, End>(std::forward<F>(f));
                }
            }
        }
    }

    namespace kernel
    {
        /**
         * @brief Assert that no arguments are passed by reference
         * @tparam T Return type of the function to check
         * @tparam Args Type of the function arguments
         **/
        template <typename T, typename... Args>
        __device__ __host__ [[nodiscard]] inline consteval bool has_reference_parameters(T (*)(Args...))
        {
            // Folds over all arguments, returning true if ANY are references
            return (std::is_reference_v<Args> || ...);
        }

        /**
         * @brief Host-side helper function to launch kernels
         * @tparam KernelFunc The kernel function to launch
         * @tparam sharedMem The amount of dynamic shared memory in bytes
         * @tparam threadBlock The size of the block as a dim3
         * @tparam Args Type of arguments to pass to the kernel
         * @param[in] grid The number of blocks to launch
         * @param[in] stream The execution stream on which to launch the kernel
         * @param[in] args Arguments to pass to the kernel
         **/
        template <const auto KernelFunc, const host::label_t sharedMem = 0, const dim3 threadBlock = dim3{block::nx<uint32_t>(), block::ny<uint32_t>(), block::nz<uint32_t>()}, typename... Args>
        __host__ inline void launch(const dim3 &grid, const cudaStream_t &stream, const Args... args) noexcept
        {
            //  Enforce that the kernel signature doesn't take references
            static_assert(!has_reference_parameters(KernelFunc), "Kernel signature cannot contain reference parameters!");

            // Launch the kernel
            KernelFunc<<<grid, threadBlock, sharedMem, stream>>>(args...);
        }
    }

    /**
     * @brief Raise a variable to a compile-time constant integer power
     * @tparam Pow The power
     * @tparam T The arithmetic type
     * @param[in] val The variable to exponent
     **/
    template <const host::label_t Pow, typename T>
    __device__ __host__ [[nodiscard]] inline constexpr T pow(const T &val) noexcept
    {
        if constexpr (Pow == 0)
        {
            return T{1};
        }
        else
        {
            return [&]<host::label_t... Is>(std::index_sequence<Is...>)
            {
                return (T{1} * ... * (static_cast<void>(Is), val));
            }(std::make_index_sequence<Pow>{});
        }
    }

    template <typename T>
    __device__ __host__ [[nodiscard]] inline constexpr T rms_sq(const T x, const T y) noexcept
    {
        return (x * x) + (y * y);
    }

    namespace GPU
    {
        /**
         * @brief Loop over all GPUs
         * @tparam F Callable type
         * @param[in] nGPUs Number of GPUs in the X, Y and Z directions
         * @param[in] f Function object to execute per iteration
         *
         * Example:
         * @code
         * GPU::forAll(nGPUs, [&](device::label_t x, y, z) {
         *     data[compute_index(bx, by, bz, tx, ty, tz)] = value;
         * });
         * @endcode
         **/
        template <typename F>
        __host__ void forAll(const host::blockLabel &nGPUs, const F &&f) noexcept
        {
            // Loops for block indices
            for (host::label_t GPU_z = 0; GPU_z < nGPUs.template value<axis::Z>(); GPU_z++)
            {
                for (host::label_t GPU_y = 0; GPU_y < nGPUs.template value<axis::Y>(); GPU_y++)
                {
                    for (host::label_t GPU_x = 0; GPU_x < nGPUs.template value<axis::X>(); GPU_x++)
                    {
                        // Execute the arbitrary loop body
                        f(GPU_x, GPU_y, GPU_z);
                    }
                }
            }
        }

        /**
         * @brief Get the current GPU device index
         * @return The index of the currently active GPU device
         **/
        __host__ [[nodiscard]] int current_ordinal() noexcept
        {
            int result = 0;

            errorHandler::handle(cudaGetDevice(&result));

            return result;
        }

        /**
         * @brief Compute a unique stream ID for a given device index
         * @param[in] deviceIdx The index of the device (GPU)
         * @return A unique stream ID for the device
         **/
        __host__ [[nodiscard]] inline constexpr host::label_t internalStreamID(const host::label_t deviceIdx) noexcept
        {
            return (deviceIdx * 3) + 1;
        }
    }

    namespace host
    {
        /**
         * @brief Nested loop over block and thread indices
         * @tparam F Type of the callable function
         * @param[in] nBlocks Number of blocks in the X, Y and Z directions
         * @param[in] f Function called for each (bx, by, bz, tx, ty, tz)
         *
         * Example:
         * @code
         * host::forAll(nBlocks, [&](device::label_t bx, by, bz, tx, ty, tz) {
         *     data[compute_index(bx, by, bz, tx, ty, tz)] = value;
         * });
         * @endcode
         **/
        template <typename F>
        __host__ void forAll(const host::blockLabel &nBlocks, const F &&f) noexcept
        {
            // Loops for block indices
            for (host::label_t bz = 0; bz < nBlocks.template value<axis::Z>(); bz++)
            {
                for (host::label_t by = 0; by < nBlocks.template value<axis::Y>(); by++)
                {
                    for (host::label_t bx = 0; bx < nBlocks.template value<axis::X>(); bx++)
                    {
                        // Loops for thread indices
                        for (host::label_t tz = 0; tz < block::nz<host::label_t>(); tz++)
                        {
                            for (host::label_t ty = 0; ty < block::ny<host::label_t>(); ty++)
                            {
                                for (host::label_t tx = 0; tx < block::nx<host::label_t>(); tx++)
                                {
                                    // Execute the arbitrary loop body
                                    f(bx, by, bz, tx, ty, tz);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    namespace global
    {
        /**
         * @brief Nested loop over global grid indices
         * @tparam F Type of the callable function
         * @param[in] dimensions Number of points in the X, Y and Z directions
         * @param[in] indent Identation to the loop end condition
         * @param[in] f Function called for each (bx, by, bz, tx, ty, tz)
         *
         * Example:
         * @code
         * global::forAll(dimensions, indent, [&](device::label_t x, y, z) {
         *     data[compute_index(bx, by, bz, tx, ty, tz)] = value;
         * });
         * @endcode
         **/
        template <typename F>
        __host__ void forAll(const host::blockLabel &dimensions, const host::blockLabel &indent, const F &&f) noexcept
        {
            for (host::label_t z = 0; z < dimensions.z - indent.z; z++)
            {
                for (host::label_t y = 0; y < dimensions.y - indent.y; y++)
                {
                    for (host::label_t x = 0; x < dimensions.x - indent.x; x++)
                    {
                        f(x, y, z);
                    }
                }
            }
        }

        /**
         * @brief Nested loop over indices in a plane
         * @tparam alpha The axis direction (X, Y or Z)
         * @tparam F Type of the callable function
         * @param[in] dimensions Number of points in the X, Y and Z directions
         * @param[in] f Function called for each (bx, by, bz, tx, ty, tz)
         *
         * Example:
         * @code
         * global::forAllInPlane(dimensions, [&](device::label_t x, y, z) {
         *     data[compute_index(bx, by, bz, tx, ty, tz)] = value;
         * });
         * @endcode
         **/
        template <const axis::type alpha, typename F>
        __host__ void forAllInPlane(const host::blockLabel &dimensions, const F &&f) noexcept
        {
            for (host::label_t j = 0; j < dimensions.value<axis::orthogonal<alpha, 1>()>(); j++)
            {
                for (host::label_t i = 0; i < dimensions.value<axis::orthogonal<alpha, 0>()>(); i++)
                {
                    f(i, j);
                }
            }
        }
    }

    /**
     * @brief Host-side indexing operations
     **/
    namespace host
    {
        /**
         * @brief Memory index (host version)
         * @param[in] tx Thread x-coordinate within block
         * @param[in] ty Thread y-coordinate within block
         * @param[in] tz Thread z-coordinate within block
         * @param[in] bx Block index in the x-direction
         * @param[in] by Block index in the y-direction
         * @param[in] bz Block index in the z-direction
         * @param[in] nxBlocks Number of blocks in x-direction
         * @param[in] nyBlocks Number of blocks in y-direction
         * @return Linearized index using mesh constants
         *
         * Layout: [bx][by][bz][tz][ty][tx] (tx fastest varying)
         **/
        __host__ [[nodiscard]] inline constexpr label_t idx(const label_t tx, const label_t ty, const label_t tz, const label_t bx, const label_t by, const label_t bz, const label_t nxBlocks, const label_t nyBlocks) noexcept
        {
            return global::idx<host::label_t, block::nx<host::label_t>(), block::ny<host::label_t>(), block::nz<host::label_t>()>(tx, ty, tz, bx, by, bz, nxBlocks, nyBlocks);
        }

        /**
         * @overload Compute the memory index from a thread and block label
         * @param[in] Tx Three-dimensional thread coordinates
         * @param[in] Bx Three-dimensional block coordinates
         * @param[in] nxBlocks Number of blocks in x-direction
         * @param[in] nyBlocks Number of blocks in y-direction
         **/
        __host__ [[nodiscard]] inline constexpr label_t idx(const host::threadLabel &Tx, const host::blockLabel &Bx, const label_t nxBlocks, const label_t nyBlocks) noexcept
        {
            return idx(Tx.x, Tx.y, Tx.z, Bx.x, Bx.y, Bx.z, nxBlocks, nyBlocks);
        }
    }

    namespace Cartesian
    {
        /**
         * @brief Global scalar field index (collapsed 3D)
         * @param[in] x Global X coordinate
         * @param[in] y Global Y coordinate
         * @param[in] z Global Z coordinate
         * @param[in] nx Global X dimension
         * @param[in] ny Global Y dimension
         * @return Linearized index: x + nx*(y + ny*z)
         **/
        __device__ __host__ [[nodiscard]] inline constexpr host::label_t idx(const host::label_t x, const host::label_t y, const host::label_t z, const host::label_t nx, const host::label_t ny) noexcept
        {
            return x + (nx * (y + (ny * z)));
        }

        /**
         * @overload
         * @param[in] point The global point coordinate
         * @param[in] nx Global X dimension
         * @param[in] ny Global Y dimension
         **/
        __device__ __host__ [[nodiscard]] inline constexpr host::label_t idx(const host::pointLabel &point, const host::label_t nx, const host::label_t ny) noexcept
        {
            return idx(point.value<axis::X>(), point.value<axis::Y>(), point.value<axis::Z>(), nx, ny);
        }
    }

    /**
     * @brief Device-side indexing operations
     **/
    namespace device
    {
        /**
         * @brief Check if current thread exceeds global bounds
         * @param[in] point The global point coordinate
         * @note Uses device constants device::nx, device::ny, device::nz
         * @return True if thread is outside domain boundaries
         **/
        __device__ [[nodiscard]] inline bool out_of_bounds(const device::pointCoordinate &point) noexcept
        {
            return ((point.value<axis::X>() >= device::n<axis::X>()) || (point.value<axis::Y>() >= device::n<axis::Y>()) || (point.value<axis::Z>() >= device::n<axis::Z>()));
        }

        /**
         * @brief Memory index (device version)
         * @param[in] tx,ty,tz Thread-local coordinates
         * @param[in] bx,by,bz Block indices
         * @return Linearized index using device constants device::NUM_BLOCK_X/Y
         *
         * Layout: [bx][by][bz][tz][ty][tx] (tx fastest varying)
         **/
        __device__ [[nodiscard]] inline device::label_t idx(
            const device::label_t tx, const device::label_t ty, const device::label_t tz,
            const device::label_t bx, const device::label_t by, const device::label_t bz) noexcept
        {
            return global::idx<device::label_t, block::nx<device::label_t>(), block::ny<device::label_t>(), block::nz<device::label_t>()>(tx, ty, tz, bx, by, bz, NUM_BLOCK_X, NUM_BLOCK_Y);
        }

        /**
         * @overload
         * @param[in] Tx Three-dimensional thread coordinates
         * @param[in] Bx Three-dimensional block coordinates
         **/
        __device__ [[nodiscard]] inline device::label_t idx(const thread::coordinate &Tx, const block::coordinate &Bx) noexcept
        {
            return idx(Tx.value<axis::X>(), Tx.value<axis::Y>(), Tx.value<axis::Z>(), Bx.value<axis::X>(), Bx.value<axis::Y>(), Bx.value<axis::Z>());
        }
    }

    namespace block
    {
        /**
         * @brief Memory index within a block (device version)
         * @param[in] tx Thread-local x-coordinate
         * @param[in] ty Thread-local y-coordinate
         * @param[in] tz Thread-local z-coordinate
         * @return Linearized index using block dimensions (block::nx() and block::ny())
         *
         * Layout within a block: [tz][ty][tx] (tx fastest varying)
         * Strides:
         *   - x-stride: 1
         *   - y-stride: block::nx()
         *   - z-stride: block::nx() * block::ny()
         **/
        __device__ [[nodiscard]] inline device::label_t idx(const device::label_t tx, const device::label_t ty, const device::label_t tz) noexcept
        {
            return tx + block::nx<device::label_t>() * (ty + block::ny<device::label_t>() * tz);
        }

        /**
         * @overload
         * @param[in] Tx Three-dimensional thread coordinates
         **/
        __device__ [[nodiscard]] inline device::label_t idx(const thread::coordinate &Tx) noexcept
        {
            return block::idx(Tx.value<axis::X>(), Tx.value<axis::Y>(), Tx.value<axis::Z>());
        }

        /**
         * @brief Wrapper for __syncthreads
         **/
        __device__ inline void sync() noexcept
        {
            __syncthreads();
        }
    }

    namespace GPU
    {
        /**
         * @brief Device index (universal version)
         * @tparam T Label type
         * @param[in] dx, dy, dz Device indices in the X, Y and Z directions
         * @param[in] ndx, ndy Number of devices in the X and Y directions
         **/
        __device__ __host__ [[nodiscard]] inline constexpr host::label_t idx(
            const host::label_t dx, const host::label_t dy, const host::label_t dz,
            const host::label_t ndx, const host::label_t ndy) noexcept
        {
            return Cartesian::idx(dx, dy, dz, ndx, ndy);
        }

        /**
         * @brief Queries a device and gets its properties
         * @param[in] deviceID The ID of the device to query
         * @return A cudaDeviceProp struct containing the properties of deviceID
         **/
        __host__ [[nodiscard]] const cudaDeviceProp properties(const int deviceID)
        {
            cudaDeviceProp props;

            errorHandler::handle(cudaGetDeviceProperties(&props, deviceID));

            return props;
        }
    }
}

#endif
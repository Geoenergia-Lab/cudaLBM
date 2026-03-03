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

namespace LBM
{
    /**
     * @brief Compile-time recursive loop unroller
     * @tparam Start Starting index (inclusive)
     * @tparam End Ending index (exclusive)
     * @tparam F Callable type accepting integral_constant<label_t>
     * @param[in] f Function object to execute per iteration
     *
     * @note Equivalent to runtime loop: `for(label_t i=Start; i<End; ++i)`
     * @note Enables `if constexpr` usage in loop bodies
     * @warning Recursion depth limited by compiler constraints
     *
     * Example usage:
     * @code
     * constexpr_for<0, 5>([](auto i) {
     *     // i is integral_constant<label_t, N>
     *     if constexpr (i.value % 2 == 0) { ... }
     * });
     * @endcode
     **/
    namespace host
    {
        template <const label_t Start, const label_t End, typename F>
        __host__ inline constexpr void constexpr_for(F &&f) noexcept
        {
            if constexpr (Start < End)
            {
                f(std::integral_constant<label_t, Start>());
                if constexpr (Start + 1 < End)
                {
                    host::constexpr_for<Start + 1, End>(std::forward<F>(f));
                }
            }
        }
    }

    namespace device
    {
        template <const label_t Start, const label_t End, typename F>
        __device__ inline constexpr void constexpr_for(F &&f) noexcept
        {
            if constexpr (Start < End)
            {
                f(integralConstant<label_t, Start>());
                if constexpr (Start + 1 < End)
                {
                    device::constexpr_for<Start + 1, End>(std::forward<F>(f));
                }
            }
        }
    }

    /**
     * @brief Raise a variable to a compile-time constant integer power
     * @tparam N The power
     * @tparam T The arithmetic type
     * @param[in] var The variable to exponent
     **/
    template <const std::size_t N, typename T>
    __device__ __host__ [[nodiscard]] inline constexpr T pow(T &&var)
    {
        using ReturnType = std::decay_t<T>;

        if constexpr (N == 0)
        {
            return ReturnType{0};
        }
        else if constexpr (N == 1)
        {
            return std::forward<T>(var);
        }
        else
        {
            return []<std::size_t... Is>(std::index_sequence<Is...>, auto &&v)
            {
                // Multiply v by itself N times
                return ((static_cast<void>(Is), v) * ...);
            }(std::make_index_sequence<N>{}, std::forward<T>(var));
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
         * GPU::forAll(nGPUs, [&](label_t x, y, z) {
         *     data[compute_index(bx, by, bz, tx, ty, tz)] = value;
         * });
         * @endcode
         **/
        template <typename F>
        __host__ void forAll(const blockLabel_t &nGPUs, const F &&f) noexcept
        {
            // Loops for block indices
            for (label_t GPU_z = 0; GPU_z < nGPUs.z; GPU_z++)
            {
                for (label_t GPU_y = 0; GPU_y < nGPUs.y; GPU_y++)
                {
                    for (label_t GPU_x = 0; GPU_x < nGPUs.x; GPU_x++)
                    {
                        // Execute the arbitrary loop body
                        f(GPU_x, GPU_y, GPU_z);
                    }
                }
            }
        }
    }

    namespace host
    {
        /**
         * @brief Nested loop over block and thread indices
         * @param[in] nBlocks Number of blocks in the X, Y and Z directions
         * @param[in] f Function called for each (bx, by, bz, tx, ty, tz)
         *
         * Example:
         * @code
         * host::forAll(nBlocks, [&](label_t bx, by, bz, tx, ty, tz) {
         *     data[compute_index(bx, by, bz, tx, ty, tz)] = value;
         * });
         * @endcode
         **/
        template <typename F>
        __host__ void forAll(const blockLabel_t &nBlocks, const F &&f) noexcept
        {
            // Loops for block indices
            for (label_t bz = 0; bz < nBlocks.z; bz++)
            {
                for (label_t by = 0; by < nBlocks.y; by++)
                {
                    for (label_t bx = 0; bx < nBlocks.x; bx++)
                    {
                        // Loops for thread indices
                        for (label_t tz = 0; tz < block::nz<label_t>(); tz++)
                        {
                            for (label_t ty = 0; ty < block::ny<label_t>(); ty++)
                            {
                                for (label_t tx = 0; tx < block::nx<label_t>(); tx++)
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
         * @param[in] dimensions Number of points in the X, Y and Z directions
         * @param[in] f Function called for each (bx, by, bz, tx, ty, tz)
         *
         * Example:
         * @code
         * global::forAll<Indent>(dimensions, [&](label_t x, y, z) {
         *     data[compute_index(bx, by, bz, tx, ty, tz)] = value;
         * });
         * @endcode
         **/
        template <const blockLabel_t Indent, typename F>
        __host__ void forAll(const blockLabel_t &dimensions, const F &&f) noexcept
        {
            for (label_t z = 0; z < dimensions.z - Indent.z; z++)
            {
                for (label_t y = 0; y < dimensions.y - Indent.y; y++)
                {
                    for (label_t x = 0; x < dimensions.x - Indent.x; x++)
                    {
                        f(x, y, z);
                    }
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
         * @param[in] tx,ty,tz Thread-local coordinates
         * @param[in] bx,by,bz Block indices
         * @param[in] nxBlocks,nyBlocks Number of blocks in the x and y directions
         * @return Linearized index using mesh constants
         *
         * Layout: [bx][by][bz][tz][ty][tx] (tx fastest varying)
         **/
        template <typename T = label_t>
        __host__ [[nodiscard]] inline constexpr T idx(const T tx, const T ty, const T tz, const T bx, const T by, const T bz, const T nxBlocks, const T nyBlocks) noexcept
        {
            return (tx + block::nx<T>() * (ty + block::ny<T>() * (tz + block::nz<T>() * (bx + nxBlocks * (by + nyBlocks * bz)))));
        }

        template <typename T = label_t>
        __host__ [[nodiscard]] inline constexpr T idx(const blockLabel_t &Tx, const blockLabel_t &Bx, const T nxBlocks, const T nyBlocks) noexcept
        {
            return idx<T>(Tx.x, Tx.y, Tx.z, Bx.x, Bx.y, Bx.z, nxBlocks, nyBlocks);
        }
    }

    namespace global
    {
        /**
         * @brief Global scalar field index (collapsed 3D)
         * @param[in] x,y,z Global coordinates
         * @param[in] nx,ny Global dimensions
         * @return Linearized index: x + nx*(y + ny*z)
         **/
        template <typename T = label_t>
        __device__ __host__ [[nodiscard]] inline T idx(const T x, const T y, const T z, const T nx, const T ny) noexcept
        {
            return x + (nx * (y + (ny * z)));
        }
    }

    /**
     * @brief Device-side indexing operations
     **/
    namespace device
    {
        /**
         * @brief Check if current thread exceeds global bounds
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
        __device__ [[nodiscard]] inline label_t idx(
            const label_t tx, const label_t ty, const label_t tz,
            const label_t bx, const label_t by, const label_t bz) noexcept
        {
            return (tx + block::nx() * (ty + block::ny() * (tz + block::nz() * (bx + NUM_BLOCK_X * (by + NUM_BLOCK_Y * bz)))));
        }

        /**
         * @overload
         * @param[in] tx Thread coordinates (thread::coordinate)
         * @param[in] bx Block indices (thread::coordinate)
         **/
        __device__ [[nodiscard]] inline label_t idx(const thread::coordinate &Tx, const block::coordinate &Bx) noexcept
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
        __device__ [[nodiscard]] inline label_t idx(const label_t tx, const label_t ty, const label_t tz) noexcept
        {
            return tx + block::nx() * (ty + block::ny() * tz);
        }

        /**
         * @overload
         * @param[in] tx Thread coordinates (thread::coordinate)
         **/
        __device__ [[nodiscard]] inline label_t idx(const thread::coordinate &Tx) noexcept
        {
            return block::idx(Tx.value<axis::X>(), Tx.value<axis::Y>(), Tx.value<axis::Z>());
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
        template <typename T = label_t>
        __device__ __host__ [[nodiscard]] inline constexpr T idx(const T dx, const T dy, const T dz, const T ndx, const T ndy) noexcept
        {
            return global::idx<T>(dx, dy, dz, ndx, ndy);
        }

        /**
         * @brief Queries a device and gets its properties
         * @param[in] deviceID The ID of the device to query
         * @return A cudaDeviceProp struct containing the properties of deviceID
         **/
        __host__ [[nodiscard]] const cudaDeviceProp properties(const int deviceID)
        {
            cudaDeviceProp props;

            errorHandler::check(cudaGetDeviceProperties(&props, deviceID));

            return props;
        }
    }

    namespace kernel
    {
        /**
         * @brief Configures a kernel function to prefer shared memory and sets its dynamic shared memory size
         * @tparam smem_alloc_size The amount of shared memory (in bytes) to allocate for the kernel
         * @tparam T The function type (e.g., a lambda or a function pointer)
         * @param[in] func The kernel function to configure
         **/
        template <const label_t smem_alloc_size, class T>
        __host__ void configure(T *func)
        {
            errorHandler::check(cudaFuncSetCacheConfig(func, cudaFuncCachePreferShared));
            errorHandler::check(cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_alloc_size));
        }
    }
}

#endif
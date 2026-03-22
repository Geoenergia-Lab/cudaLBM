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
    Handles the use of the cache on the GPU

Namespace
    LBM::cache

SourceFiles
    cache.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_CACHE_CUH
#define __MBLBM_CACHE_CUH

#include "../LBMIncludes.cuh"
#include "../typedefs/typedefs.cuh"

namespace LBM
{
    namespace cache
    {
        /**
         * @namespace Policy
         * @brief Enumerates cache eviction policies for prefetch operations
         **/
        namespace Policy
        {
            /**
             * @enum Enum
             * @brief Cache eviction policy options
             **/
            typedef enum Enum : device::label_t
            {
                evict_first = 0,
                evict_last = 1
            } Enum;
        }

        /**
         * @namespace Level
         * @brief Enumerates cache hierarchy levels for prefetch operations
         **/
        namespace Level
        {
            /**
             * @enum Enum
             * @brief Cache level options
             **/
            typedef enum Enum : device::label_t
            {
                L1 = 0,
                L2 = 1
            } Enum;
        }

        /**
         * @brief Prefetch data to a specific cache level with specified eviction policy
         * @tparam level Cache level to prefetch to (Level::L1 or Level::L2)
         * @tparam policy Cache eviction policy (Policy::evict_first or Policy::evict_last)
         * @tparam T Type of data being prefetched
         * @param[in] ptr Pointer to data to be prefetched
         *
         * This function uses CUDA inline assembly to issue hardware prefetch instructions
         * that move data into the specified cache level before it's needed. This can
         * significantly reduce memory latency for carefully orchestrated memory access patterns.
         *
         * @note Requires CUDA architecture >= 350 (Kepler or newer)
         * @note Uses restrict qualifier to indicate no pointer aliasing
         * @note Compile-time validation ensures only valid cache levels and policies are used
         **/
        template <const Level::Enum level, const Policy::Enum policy, typename T>
        __device__ inline void prefetch(const T *const ptrRestrict ptr) noexcept
        {
            // Check that the CUDA architecture is valid
#if defined(__CUDA_ARCH__)
            static_assert((__CUDA_ARCH__ >= 350), "CUDA architecture must be >= 350");
#endif

            // Check that the cache level is valid
            static_assert((level == Level::L1) | (level == Level::L2), "Prefetch cache level must be 1 or 2");

            // Check that the eviction policy is valid
            static_assert((policy == Policy::evict_first) | (policy == Policy::evict_last), "Cache eviction policy must be evict_first or evict_last");

            if constexpr (level == Level::L1)
            {
                if constexpr (policy == Policy::evict_first)
                {
#if (__CUDA_ARCH__ >= 800)
                    asm volatile("prefetch.global.L1::evict_first [%0];" ::"l"(ptr));
#else
                    asm volatile("prefetch.global.L1 [%0];" ::"l"(ptr));
#endif
                }
                else if constexpr (policy == Policy::evict_last)
                {
#if (__CUDA_ARCH__ >= 800)
                    asm volatile("prefetch.global.L1::evict_last [%0];" ::"l"(ptr));
#else
                    asm volatile("prefetch.global.L1 [%0];" ::"l"(ptr));
#endif
                }
            }
            else if constexpr (level == Level::L2)
            {
                if constexpr (policy == Policy::evict_first)
                {
#if (__CUDA_ARCH__ >= 800)
                    asm volatile("prefetch.global.L2::evict_first [%0];" ::"l"(ptr));
#else
                    asm volatile("prefetch.global.L2 [%0];" ::"l"(ptr));
#endif
                }
                else if constexpr (policy == Policy::evict_last)
                {
#if (__CUDA_ARCH__ >= 800)
                    asm volatile("prefetch.global.L2::evict_last [%0];" ::"l"(ptr));
#else
                    asm volatile("prefetch.global.L1 [%0];" ::"l"(ptr));
#endif
                }
            }
        }
    }
}

#endif
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
    Compile-time constants for the GPU

Namespace
    LBM, LBM::block

SourceFiles
    blockConfig.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_BLOCKCONFIG_CUH
#define __MBLBM_BLOCKCONFIG_CUH

namespace LBM
{
    /**
     * @brief CUDA block dimension configuration
     * @details Compile-time constants defining thread block dimensions
     **/
    namespace block
    {
        /**
         * @brief Threads per block in x-dimension (compile-time constant)
         **/
        template <typename T = device::label_t>
        __device__ __host__ [[nodiscard]] inline consteval T nx() noexcept
        {
#ifdef SCALAR_PRECISION
            types::assertions::validate<scalar_t>();

            return 8 * sizeof(float) / (sizeof(scalar_t));
#else
            return 8;
#endif
        }

        /**
         * @brief Threads per block in y-dimension (compile-time constant)
         **/
        template <typename T = device::label_t>
        __device__ __host__ [[nodiscard]] inline consteval T ny() noexcept
        {
#ifdef SCALAR_PRECISION
            types::assertions::validate<scalar_t>();

            return 8 * sizeof(float) / (sizeof(scalar_t));
#else
            return 8;
#endif
        }

        /**
         * @brief Threads per block in z-dimension (compile-time constant)
         **/
        template <typename T = device::label_t>
        __device__ __host__ [[nodiscard]] inline consteval T nz() noexcept
        {
#ifdef SCALAR_PRECISION
            types::assertions::validate<scalar_t>();

            return 8 * sizeof(float) / (sizeof(scalar_t));
#else
            return 8;
#endif
        }

        /**
         * @brief Threads per block in an arbitrary dimension (compile-time constant)
         **/
        template <axis::type alpha, typename T = device::label_t>
        __device__ __host__ [[nodiscard]] inline consteval T n() noexcept
        {
            axis::assertions::validate<alpha, axis::NOT_NULL>();

            return var3<T>(block::nx<T>(), block::ny<T>(), block::nz<T>()).template value<alpha>();
        }

        /**
         * @brief Total threads per block (nx * ny * nz)
         **/
        template <typename T = device::label_t>
        __device__ __host__ [[nodiscard]] inline consteval T size() noexcept
        {
            return nx<T>() * ny<T>() * nz<T>();
        }

        /**
         * @brief Padding for the shared memory
         **/
        template <typename T = device::label_t>
        __device__ __host__ [[nodiscard]] inline consteval T padding() noexcept
        {
            return 33;
        }

        /**
         * @brief Stride for the shared memory
         **/
        template <typename T = device::label_t>
        __device__ __host__ [[nodiscard]] inline consteval T stride() noexcept
        {
            return size<T>() + padding<T>();
        }

        /**
         * @brief Linear stride in the z-direction for global x-major memory layout
         **/
        __device__ __host__ [[nodiscard]] inline consteval device::label_t stride_z() noexcept
        {
            return nx() * ny();
        }

        /**
         * @brief Total size of the shared memory
         **/
        template <class VelocitySet, const host::label_t nVars>
        __device__ __host__ [[nodiscard]] inline consteval host::label_t sharedMemoryBufferSize(const host::label_t variableSize = 1) noexcept
        {
            constexpr const host::label_t A = (VelocitySet::template Q<host::label_t>() - 1) * block::stride<host::label_t>();
            constexpr const host::label_t B = block::size<host::label_t>() * (nVars + 1);
            return (A > B ? A : B) * variableSize;
        }

        /**
         * @brief Size of the warp (32)
         **/
        __device__ __host__ [[nodiscard]] inline consteval device::label_t warp_size() noexcept
        {
            return 32;
        }

        /**
         * @brief Launch bounds information
         * @note These variables are device specific - enable modification later
         **/
        __host__ [[nodiscard]] inline consteval device::label_t maxThreads() noexcept
        {
            return block::nx() * block::ny() * block::nz();
        }
    }
}

#endif
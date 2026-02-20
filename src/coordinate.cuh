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
    A list of integral typedefs used throughout the cudaLBM source code

Namespace
    LBM

SourceFiles
    coordinateTypedefs.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_COORDINATETYPEDEFS_CUH
#define __MBLBM_COORDINATETYPEDEFS_CUH

#include "globalConstants.cuh"

namespace LBM
{
    namespace thread
    {
        class coordinate
        {
        public:
            /**
             * @brief Constructs from threadIdx
             **/
            __device__ [[nodiscard]] inline explicit coordinate() noexcept
                : x(static_cast<label_t>(threadIdx.x)),
                  y(static_cast<label_t>(threadIdx.y)),
                  z(static_cast<label_t>(threadIdx.z)) {}

            /**
             * @brief Returns the ordinate in a particular axis
             * @tparam alpha The axis
             **/
            template <axis::type alpha>
            __device__ __host__ [[nodiscard]] inline constexpr label_t value() const noexcept
            {
                axis::assertions::validate<alpha, axis::NOT_NULL>();

                if constexpr (alpha == axis::X)
                {
                    return x;
                }

                if constexpr (alpha == axis::Y)
                {
                    return y;
                }

                if constexpr (alpha == axis::Z)
                {
                    return z;
                }
            }

            /**
             * @brief Shifts the coordinate along a particular axis by a coefficient
             * @tparam alpha The axis
             * @tparam coeff The coefficient to shift by (-1, 0 or +1)
             **/
            template <axis::type alpha, const int coeff>
            __device__ [[nodiscard]] inline constexpr label_t shifted_coordinate() const noexcept
            {
                axis::assertions::validate<alpha, axis::NOT_NULL>();

                velocityCoefficient::assertions::validate<coeff, velocityCoefficient::NOT_NULL>();

                if constexpr (coeff == -1)
                {
                    return (value<alpha>() - 1 + block::n<alpha>()) % block::n<alpha>();
                }

                if constexpr (coeff == 0)
                {
                    return value<alpha>();
                }

                if constexpr (coeff == 1)
                {
                    return (value<alpha>() + 1 + block::n<alpha>()) % block::n<alpha>();
                }
            }

        private:
            /**
             * @brief The underlying thread coordinates
             **/
            const label_t x;
            const label_t y;
            const label_t z;
        };
    }

    namespace block
    {
        class coordinate
        {
        public:
            /**
             * @brief Constructs from blockIdx
             **/
            __device__ [[nodiscard]] inline explicit coordinate() noexcept
                : x(static_cast<label_t>(blockIdx.x)),
                  y(static_cast<label_t>(blockIdx.y)),
                  z(static_cast<label_t>(blockIdx.z)) {}

            /**
             * @brief Returns the ordinate in a particular axis
             * @tparam alpha The axis
             **/
            template <axis::type alpha>
            __device__ __host__ [[nodiscard]] inline constexpr label_t value() const noexcept
            {
                axis::assertions::validate<alpha, axis::NOT_NULL>();

                if constexpr (alpha == axis::X)
                {
                    return x;
                }

                if constexpr (alpha == axis::Y)
                {
                    return y;
                }

                if constexpr (alpha == axis::Z)
                {
                    return z;
                }
            }

            /**
             * @brief Shifts the coordinate along a particular axis by a coefficient
             * @tparam alpha The axis
             * @tparam coeff The coefficient to shift by (-1, 0 or +1)
             **/
            template <const axis::type alpha, const int coeff>
            __device__ [[nodiscard]] inline constexpr label_t shifted_block() const noexcept
            {
                axis::assertions::validate<alpha, axis::NOT_NULL>();

                velocityCoefficient::assertions::validate<coeff, velocityCoefficient::NOT_NULL>();

                if constexpr (coeff == -1)
                {
                    return (value<alpha>() - 1 + device::NUM_BLOCK<alpha>()) % device::NUM_BLOCK<alpha>();
                }

                if constexpr (coeff == 0)
                {
                    return value<alpha>();
                }

                if constexpr (coeff == 1)
                {
                    return (value<alpha>() + 1 + device::NUM_BLOCK<alpha>()) % device::NUM_BLOCK<alpha>();
                }
            }

        private:
            /**
             * @brief The underlying block coordinates
             **/
            const label_t x;
            const label_t y;
            const label_t z;
        };
    }

    namespace device
    {
        class pointCoordinate
        {
        public:
            /**
             * @brief Constructs from thread and block coordinates
             * @param[in] Tx The thread coordinates
             * @param[in] Bx The block coordinates
             **/
            __device__ [[nodiscard]] inline explicit pointCoordinate(
                const thread::coordinate &Tx,
                const block::coordinate &Bx) noexcept
                : x(Tx.value<axis::X>() + block::nx<label_t>() * (Bx.value<axis::X>() + device::BLOCK_OFFSET_X)),
                  y(Tx.value<axis::Y>() + block::ny<label_t>() * (Bx.value<axis::Y>() + device::BLOCK_OFFSET_Y)),
                  z(Tx.value<axis::Z>() + block::nz<label_t>() * (Bx.value<axis::Z>() + device::BLOCK_OFFSET_Z)) {}

            /**
             * @brief Returns the ordinate in a particular axis
             * @tparam alpha The axis
             **/
            template <axis::type alpha>
            __device__ __host__ [[nodiscard]] inline constexpr label_t value() const noexcept
            {
                axis::assertions::validate<alpha, axis::NOT_NULL>();

                if constexpr (alpha == axis::X)
                {
                    return x;
                }

                if constexpr (alpha == axis::Y)
                {
                    return y;
                }

                if constexpr (alpha == axis::Z)
                {
                    return z;
                }
            }

        private:
            /**
             * @brief The underlying point coordinates
             **/
            const label_t x;
            const label_t y;
            const label_t z;
        };
    }
}

#endif
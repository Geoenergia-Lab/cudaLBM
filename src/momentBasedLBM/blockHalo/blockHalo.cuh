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
    Top-level header file for the halo class

Namespace
    LBM::device

SourceFiles
    blockHalo.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_BLOCKHALO_CUH
#define __MBLBM_BLOCKHALO_CUH

namespace LBM
{
    namespace host
    {
        /**
         * @brief Index for arbitrarily aligned population arrays
         * @tparam QF Number of populations
         * @tparam alpha The axis direction (X, Y or Z)
         * @param[in] pop Population index
         * @param[in] Tx Three-dimensional thread coordinates
         * @param[in] Bx Three-dimensional block coordinates
         * @param[in] nxBlocks Number of blocks in x-direction
         * @param[in] nyBlocks Number of blocks in y-direction
         * @return Linearized index: idxPop<alpha>
         **/
        template <const axis::type alpha, const label_t QF>
        __host__ [[nodiscard]] inline label_t idxPop(const label_t pop, const threadLabel &Tx, const blockLabel &Bx, const label_t nxBlocks, const label_t nyBlocks) noexcept
        {
            return Tx.value<axis::orthogonal<alpha, 0>()>() + block::n<axis::orthogonal<alpha, 0>()>() * (Tx.value<axis::orthogonal<alpha, 1>()>() + block::n<axis::orthogonal<alpha, 1>()>() * (pop + QF * (Bx.x + nxBlocks * (Bx.y + nyBlocks * Bx.z))));
        }
    }

    namespace device
    {
        /**
         * @brief Population index for alpha-aligned arrays (device version)
         * @tparam alpha The axis direction (X, Y or Z)
         * @tparam pop Population index
         * @tparam QF Number of populations
         * @param[in] ta Thread-local alpha coordinate
         * @param[in] tb Thread-local beta coordinate
         * @param[in] bx Block index in the x-direction
         * @param[in] by Block index in the y-direction
         * @param[in] bz Block index in the z-direction
         * @return Linearized two-dimensional face index
         **/
        template <const axis::type alpha, const device::label_t pop, const device::label_t QF>
        __device__ [[nodiscard]] inline device::label_t idxPop(
            const device::label_t ta,
            const device::label_t tb,
            const device::label_t bx,
            const device::label_t by,
            const device::label_t bz) noexcept
        {
            axis::assertions::validate<alpha, axis::NOT_NULL>();

            return ta + block::n<axis::orthogonal<alpha, 0>()>() * (tb + block::n<axis::orthogonal<alpha, 1>()>() * (pop + QF * (bx + device::NUM_BLOCK_X * (by + device::NUM_BLOCK_Y * bz))));
        }

        /**
         * @overload
         * @param[in] Bx Three-dimensional block coordinates
         **/
        template <const axis::type alpha, const device::label_t pop, const device::label_t QF>
        __device__ [[nodiscard]] inline device::label_t idxPop(
            const device::label_t talpha, const device::label_t tbeta,
            const block::coordinate &Bx) noexcept
        {
            return idxPop<alpha, pop, QF>(talpha, tbeta, Bx.value<axis::X>(), Bx.value<axis::Y>(), Bx.value<axis::Z>());
        }

        /**
         * @overload
         * @param[in] ij The thread-local 2D coordinate within the block
         * @param[in] Bx Three-dimensional block coordinates
         **/
        template <const axis::type alpha, const device::label_t pop, const device::label_t QF>
        __device__ [[nodiscard]] inline device::label_t idxPop(
            const dim2 &ij,
            const block::coordinate &Bx) noexcept
        {
            return idxPop<alpha, pop, QF>(ij.i(), ij.j(), Bx.value<axis::X>(), Bx.value<axis::Y>(), Bx.value<axis::Z>());
        }

        /**
         * @brief Returns the pointer index corresponding to the axis direction alpha and coefficient coeff (must be -1 or 1)
         * @tparam alpha The axis direction (X, Y or Z)
         * @tparam coeff The coefficient indicating the direction along the axis (must be -1 or 1)
         * @returns The pointer index corresponding to the axis direction alpha and coefficient coeff
         **/
        template <const axis::type alpha, const int coeff>
        __device__ __host__ [[nodiscard]] static inline consteval axis::pointerIndex_t pointerIndex() noexcept
        {
            axis::assertions::validate<alpha, axis::null::NOT_NULL>();
            velocityCoefficient::assertions::validate<coeff, velocityCoefficient::NOT_NULL>();

            if constexpr (alpha == axis::X)
            {
                if constexpr (coeff == -1)
                {
                    return axis::pointerIndex_t::West;
                }
                if constexpr (coeff == 1)
                {
                    return axis::pointerIndex_t::East;
                }
            }

            if constexpr (alpha == axis::Y)
            {
                if constexpr (coeff == -1)
                {
                    return axis::pointerIndex_t::South;
                }
                if constexpr (coeff == 1)
                {
                    return axis::pointerIndex_t::North;
                }
            }

            if constexpr (alpha == axis::Z)
            {
                if constexpr (coeff == -1)
                {
                    return axis::pointerIndex_t::Back;
                }
                if constexpr (coeff == 1)
                {
                    return axis::pointerIndex_t::Front;
                }
            }
        }
    }
}

#include "halo.cuh"
#include "deviceHalo.cuh"

#endif
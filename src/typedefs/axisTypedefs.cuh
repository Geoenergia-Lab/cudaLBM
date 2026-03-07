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
    A list of typedefs used throughout the cudaLBM source code

Namespace
    LBM

SourceFiles
    axisTypedefs.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_AXISTYPEDEFS_CUH
#define __MBLBM_AXISTYPEDEFS_CUH

namespace LBM
{
    namespace axis
    {
        /**
         * @brief Cardinal axis directions: X, Y, Z or NO_DIRECTION
         **/
        typedef enum Enum : int
        {
            NO_DIRECTION = -1,
            X = 0,
            Y = 1,
            Z = 2,
        } type;

        /**
         * @brief Enumerated type for indexing pointers to halos
         **/
        typedef enum pointerIndexEnum : label_t
        {
            West = 0,
            East = 1,
            South = 2,
            North = 3,
            Back = 4,
            Front = 5
        } pointerIndex_t;

        /**
         * @brief Returns axis directions orthogonal to alpha
         * @tparam alpha The axis direction
         * @tparam i The index of the orthogonal axis (must be 0 or 1)
         * @returns One of two axis directions orthogonal to alpha
         **/
        template <const axis::type alpha, const label_t i>
        __device__ __host__ [[nodiscard]] inline consteval axis::type orthogonal() noexcept
        {
            static_assert(i < 2, "Index of axis orthogonal to alpha must be < 2");

            if constexpr (alpha == axis::X)
            {
                if constexpr (i == 0)
                {
                    return axis::Y;
                }
                if constexpr (i == 1)
                {
                    return axis::Z;
                }
            }

            if constexpr (alpha == axis::Y)
            {
                if constexpr (i == 0)
                {
                    return axis::X;
                }
                if constexpr (i == 1)
                {
                    return axis::Z;
                }
            }

            if constexpr (alpha == axis::Z)
            {
                if constexpr (i == 0)
                {
                    return axis::X;
                }
                if constexpr (i == 1)
                {
                    return axis::Y;
                }
            }
        }

        /**
         * @brief Enumerated type for axes: The axis either can or cannot be null
         **/
        typedef enum nullEnum : bool
        {
            NOT_NULL = false,
            CAN_BE_NULL = true
        } null;

        namespace assertions
        {
            /**
             * @brief Asserts that the direction alpha is a valid axis direction
             * @tparam alpha The axis direction
             * @tparam potentialNull Switch that determines whether alpha is allowed to be NO_DIRECTION or not
             **/
            template <const LBM::axis::type alpha, const LBM::axis::null null>
            __device__ __host__ inline consteval void validate() noexcept
            {
                if constexpr (null == LBM::axis::CAN_BE_NULL)
                {
                    static_assert(((alpha == LBM::axis::X) || (alpha == LBM::axis::Y) || (alpha == LBM::axis::Z) || (alpha == LBM::axis::NO_DIRECTION)), "Axis direction must be X, Y or Z");
                }
                else
                {
                    static_assert(((alpha == LBM::axis::X) || (alpha == LBM::axis::Y) || (alpha == LBM::axis::Z)), "Axis direction must be X, Y, Z or NO_DIRECTION");
                }
            }
        }
    }
}

#endif
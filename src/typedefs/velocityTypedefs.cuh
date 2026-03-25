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
    velocityTypedefs.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_VELOCITYTYPEDEFS_CUH
#define __MBLBM_VELOCITYTYPEDEFS_CUH

namespace LBM
{
    namespace velocityCoefficient
    {
        /**
         * @brief Enumerated type for velocity coefficients: The coefficient either can or cannot be null
         **/
        typedef enum nullEnum : bool
        {
            NOT_NULL = false,
            CAN_BE_NULL = true
        } null;

        namespace assertions
        {
            /**
             * @brief Asserts that coeff is a valid velocity set coefficient
             * @tparam coeff The velocity set coefficient
             **/
            template <const int coeff, const null Null>
            __device__ __host__ inline consteval void validate() noexcept
            {
                if constexpr (Null == CAN_BE_NULL)
                {
                    static_assert(((coeff == 0) || (coeff == -1) || (coeff == 1)), "Coeff must be -1, 0 or +1.");
                }
                else
                {
                    static_assert(((coeff == -1) || (coeff == 1)), "Coeff must be -1 or +1.");
                }
            }
        }
    }
}

#endif
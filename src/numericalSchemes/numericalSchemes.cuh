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
    Top-level header file for the numerical schemes library

Namespace
    LBM

SourceFiles
    numericalSchemes.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_NUMERICALSCHEMES_CUH
#define __MBLBM_NUMERICALSCHEMES_CUH

#include "../LBMIncludes.cuh"
#include "../typedefs/typedefs.cuh"

namespace LBM
{
    namespace numericalSchemes
    {
        namespace assertions
        {
            /**
             * @brief Check that the selected numerical scheme order is valid: positive multiple number of 2 up to the maximum scheme order
             **/
            template <const device::label_t Order, const device::label_t MaximumSchemeOrder>
            __device__ __host__ inline consteval void validate() noexcept
            {
                static_assert(((Order % 2 == 0) && (Order < (1 << MaximumSchemeOrder))), "Invalid numerical scheme order");
            }
        }

        /**
         * @brief Calculates the magnitude of a 3D vector field.
         * @tparam T The data type of the vector components.
         * @param[in] u A vector representing the x-components of the vector field.
         * @param[in] v A vector representing the y-components of the vector field.
         * @param[in] w A vector representing the z-components of the vector field.
         * @return A vector containing the magnitude of the vector field at each point.
         **/
        template <typename T>
        __host__ [[nodiscard]] const std::vector<T> mag(const std::vector<T> &u, const std::vector<T> &v, const std::vector<T> &w)
        {
            // Add a size check here

            std::vector<scalar_t> magu(u.size(), 0);

            for (device::label_t i = 0; i < u.size(); i++)
            {
                magu[i] = std::sqrt((u[i] * u[i]) + (v[i] * v[i]) + (w[i] * w[i]));
            }

            return magu;
        }
    }
}

#include "derivativeSchemes/derivativeSchemes.cuh"
#include "integrationSchemes/fieldIntegrate.cuh"

#endif
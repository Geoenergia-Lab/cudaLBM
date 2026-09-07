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
             * @tparam Order Order of the numerical scheme
             * @tparam MaximumSchemeOrder Maximum permissible order
             **/
            template <const host::label_t Order, const host::label_t MaximumSchemeOrder>
            __device__ __host__ inline consteval void validate() noexcept
            {
                static_assert(((Order % 2 == 0) && (Order <= MaximumSchemeOrder)), "Invalid numerical scheme order");
            }
        }

        /**
         * @brief Indicates whether a field extremum is computed on raw or absolute values.
         **/
        typedef enum absModeEnum : bool
        {
            SIGNED = false, // Use signed (raw) values.
            ABS = true      // Use absolute values.
        } absMode;

        /**
         * @brief Indicates whether or not a term is to be squared
         **/
        typedef enum sqModeEnum : bool
        {
            NOT_SQUARED = false, // Use non-squared values.
            SQUARED = true       // Use squared values.
        } sqMode;

        /**
         * @brief Calculates the magnitude of a tensor of arbitrary rank at a given index
         * @tparam Squared Calculate the square magnitude
         * @tparam T The data type of the vector components.
         * @param[in] f The tensor
         * @param[in] i The index
         * @return The magnitude of the tensor at the given index
         **/
        template <const sqMode Squared, typename T>
        __host__ [[nodiscard]] T mag(const std::vector<std::vector<T>> &f, const host::label_t i)
        {
            // Do the accumulation of the magnitude in double precision
            double result = static_cast<double>(0);
            for (host::label_t field = 0; field < f.size(); field++)
            {
                const double component = static_cast<double>(f[field][i]);
                result = result + (component * component);
            }

            if constexpr (Squared == SQUARED)
            {
                return static_cast<T>(result);
            }
            else
            {
                return static_cast<T>(std::sqrt(result));
            }
        }

        /**
         * @brief Calculates the magnitude of a tensor of arbitrary rank
         * @tparam Squared Calculate the square magnitude
         * @tparam T The data type of the vector components.
         * @param[in] f The tensor
         * @return The magnitude of the tensor
         **/
        template <const sqMode Squared, typename T>
        __host__ [[nodiscard]] const std::vector<T> mag(const std::vector<std::vector<T>> &f)
        {
            std::vector<T> vec_result(f[0].size(), 0);

            for (host::label_t i = 0; i < f[0].size(); i++)
            {
                // Cast the result back to the desired type
                vec_result[i] = mag<Squared>(f, i);
            }

            return vec_result;
        }

    }
}

#include "interpolationSchemes.cuh"
#include "derivativeSchemes/derivativeSchemes.cuh"
#include "integrationSchemes/fieldIntegrate.cuh"

#endif
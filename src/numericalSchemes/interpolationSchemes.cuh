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
    Top-level header file for the interpolation schemes library

Namespace
    LBM

SourceFiles
    interpolationSchemes.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_INTERPOLATIONSCHEMES_CUH
#define __MBLBM_INTERPOLATIONSCHEMES_CUH

namespace LBM
{
    namespace numericalSchemes
    {
        /**
         * @brief Class for performing interpolation of field values at arbitrary points in the mesh
         * @tparam T The type of the field values and interpolation weight (e.g., scalar_t)
         **/
        template <typename T>
        class interpolate
        {
        public:
            /**
             * @brief Construct an interpolate object with the field values at the two nearest grid points and the interpolation weight
             * @param[in] f0 The field value at the first grid point
             * @param[in] f1 The field value at the second grid point
             * @param[in] weight The interpolation weight, which should be between 0 and 1, where 0 corresponds to f0 and 1 corresponds to f1
             *
             **/
            __host__ [[nodiscard]] inline constexpr interpolate(const T f0, const T f1, const T weight) noexcept
                : f0_(f0),
                  f1_(f1),
                  weight_(weight) {}

            /**
             * @brief Perform linear interpolation using the stored field values and weight
             * @tparam ValueType The type of the field values and interpolation weight (e.g., scalar_t)
             * @param[in] f_0 The field value at the first grid point
             * @param[in] f_1 The field value at the second grid point
             * @param[in] W The interpolation weight, which should be between 0 and 1, where 0 corresponds to f0 and 1 corresponds to f1
             * @return The interpolated field value at the arbitrary point
             **/
            template <typename ValueType>
            __host__ [[nodiscard]] static inline constexpr ValueType linear(const ValueType f_0, const ValueType f_1, const ValueType W) noexcept
            {
                return ((static_cast<ValueType>(1) - W) * f_0) + (W * f_1);
            }

            /**
             * @brief Perform linear interpolation using the stored field values and weight
             * @return The interpolated field value at the arbitrary point
             **/
            __host__ [[nodiscard]] inline constexpr T linear() const noexcept
            {
                return interpolate::linear(f0_, f1_, weight_);
            }

        private:
            /**
             * @brief The field values at the two nearest grid points and the interpolation weight for performing interpolation at an arbitrary point
             **/
            const T f0_;
            const T f1_;
            const T weight_;
        };
    }
}

#endif
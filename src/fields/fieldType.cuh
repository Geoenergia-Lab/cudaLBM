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
    Class storing information about the type of the solution field

Namespace
    LBM

SourceFiles
    fieldType.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_FIELDINFO_CUH
#define __MBLBM_FIELDINFO_CUH

namespace LBM
{
    template <const host::label_t N>
    class fieldType
    {
    public:
        /**
         * @brief Constructs the field from a name
         * @param[in] name Name of the field
         **/
        __host__ [[nodiscard]] inline constexpr fieldType(const std::string &name) noexcept : name_(name) {}

        /**
         * @brief Helper function to generate component names based on the base name and the number of components (N).
         * @tparam ReturnType The type of the returned collection (e.g., std::array<std::string, N> or std::vector<std::string>).
         * @param[in] baseName The base name for the field, used to generate component names.
         * @return A collection of component names corresponding to the field components, following a consistent naming convention based on N.
         **/
        template <class ReturnType>
        __host__ [[nodiscard]] static const ReturnType makeComponentNames(const name_t &baseName)
        {
            static_assert(N == 1 || N == 3 || N == 6, "Unsupported component count");

            if constexpr (N == 1)
            {
                return {baseName};
            }

            if constexpr (N == 3)
            {
                return {baseName + "_x", baseName + "_y", baseName + "_z"};
            }

            if constexpr (N == 6)
            {
                return {baseName + "_xx", baseName + "_xy", baseName + "_xz", baseName + "_yy", baseName + "_yz", baseName + "_zz"};
            }
        }

        /**
         * @brief Provides read-only access to the name of the field
         **/
        __host__ [[nodiscard]] inline constexpr const name_t &name() const noexcept
        {
            return name_;
        }

    private:
        /**
         * @brief Name of the field
         **/
        const name_t name_;
    };

    template <const time::type TimeType>
    class timeType
    {
    public:
        /**
         * @brief Returns the time type of the array.
         * @return time::type value (instantaneous or time‑averaged).
         **/
        __host__ [[nodiscard]] static inline consteval time::type type() noexcept
        {
            return TimeType;
        }
    };
}

#endif

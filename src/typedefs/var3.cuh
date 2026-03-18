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
    A list of aggregate typedefs used throughout the cudaLBM source code.

Namespace
    LBM

SourceFiles
    var3.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_AGGREGATETYPEDEFS_CUH
#define __MBLBM_AGGREGATETYPEDEFS_CUH

namespace LBM
{
    /**
     * @brief Three‑component vector template
     * @tparam T Numeric type of the components
     */
    template <typename T>
    struct var3
    {
    public:
        using value_type = T;

        /**
         * @brief Public members
         */
        const T x;
        const T y;
        const T z;

        /**
         * @brief Constructor
         * @param[in] X, Y, Z Initialiser values
         */
        __host__ __device__ [[nodiscard]] inline constexpr var3(const T X, const T Y, const T Z) noexcept
            : x(X),
              y(Y),
              z(Z) {}

        /**
         * @brief Access the data by axis
         * @tparam alpha The axis (X, Y or Z)
         */
        template <axis::type alpha, typename ValueType = value_type>
        __host__ __device__ [[nodiscard]] constexpr ValueType value() const noexcept
        {
            axis::assertions::validate<alpha, axis::NOT_NULL>();

            if constexpr (alpha == axis::X)
            {
                return static_cast<ValueType>(x);
            }

            if constexpr (alpha == axis::Y)
            {
                return static_cast<ValueType>(y);
            }

            if constexpr (alpha == axis::Z)
            {
                return static_cast<ValueType>(z);
            }
        }

        /**
         * @brief Print the structure to an output stream
         * @param[in] name Name to identify the structure in the output
         * @param[in] os Output stream to write to
         */
        void print(const name_t &name, std::ostream &os) const noexcept
        {
            os << name << std::endl;
            os << "{" << std::endl;
            os << "    x = " << x << ";" << std::endl;
            os << "    y = " << y << ";" << std::endl;
            os << "    z = " << z << ";" << std::endl;
            os << "};" << std::endl;
        }

        /**
         * @brief Print the structure to std::cout
         * @param[in] name Name to identify the structure in the output
         **/
        void print(const name_t &name) const noexcept
        {
            print(name, std::cout);
        }
    };

    /**
     * @brief Block dimensions descriptor (specialisation for device::label_t)
     */
    struct blockLabel : var3<device::label_t>
    {
    public:
        /**
         * @brief Inherit constructors
         */
        using var3<value_type>::var3;

        /**
         * @brief Total size
         */
        template <typename ValueType = value_type>
        __host__ __device__ [[nodiscard]] inline constexpr ValueType size() const noexcept
        {
            return value<axis::X, ValueType>() * value<axis::Y, ValueType>() * value<axis::Z, ValueType>();
        }
    };

    using pointLabel = blockLabel;
    using threadLabel = blockLabel;

    /**
     * @brief Generic vector of scalar_t
     */
    struct pointVector : var3<scalar_t>
    {
    public:
        using var3<value_type>::var3;
    };

} // namespace LBM

#endif
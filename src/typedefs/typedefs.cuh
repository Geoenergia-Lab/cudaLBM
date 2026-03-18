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
    LBMTypedefs.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_TYPEDEFS_CUH
#define __MBLBM_TYPEDEFS_CUH

#include "../LBMIncludes.cuh"

namespace LBM
{
    /**
     * @brief Shorthand for __restrict__
     **/
#define ptrRestrict __restrict__

    /**
     * @brief Shorthand for name_t and words_t
     **/
    typedef std::string name_t;
    typedef std::vector<name_t> words_t;

    /**
     * @brief Endianness of the system: big or little
     **/
    namespace endian
    {
        typedef enum Enum : bool
        {
            BIG = false,
            LITTLE = true
        } type;

        /**
         * @brief Validate the endianness of the file system
         **/
        namespace assertions
        {
            __device__ __host__ inline consteval void validate() noexcept
            {
                static_assert(((std::endian::native == std::endian::little) || (std::endian::native == std::endian::big)), "System must be little or big endian");
            }
        }

        /**
         * @brief Get the string that corresponds to the endian type
         **/
        __device__ __host__ inline consteval const char *nameString() noexcept
        {
            assertions::validate();

            if constexpr (std::endian::native == std::endian::little)
            {
                return "littleEndian";
            }

            if constexpr (std::endian::native == std::endian::big)
            {
                return "bigEndian";
            }
        }
    }

    /**
     * @brief Time stepping types: instantaneous or time-averaged
     **/
    namespace time
    {
        typedef enum Enum : bool
        {
            timeAverage = false,
            instantaneous = true
        } type;

        /**
         * @brief Validate the endianness of the file system
         **/
        namespace assertions
        {
            template <const type TimeType>
            __device__ __host__ inline consteval void validate() noexcept
            {
                static_assert(((TimeType == timeAverage) || (TimeType == instantaneous)), "Time step type must be instantaneous or timeAverage");
            }
        }

        template <const type TimeType>
        __device__ __host__ inline consteval const char *nameString() noexcept
        {
            assertions::validate<TimeType>();

            if constexpr (TimeType == timeAverage)
            {
                return "timeAverage";
            }

            if constexpr (TimeType == instantaneous)
            {
                return "instantaneous";
            }
        }
    }
}

#include "integralTypedefs.cuh"
#include "arithmeticTypedefs.cuh"

namespace LBM
{
    /**
     * @brief Enumerated variable indices
     **/
    namespace index
    {
        typedef enum Enum : host::label_t
        {
            rho = 0,
            u = 1,
            v = 2,
            w = 3,
            xx = 4,
            xy = 5,
            xz = 6,
            yy = 7,
            yz = 8,
            zz = 9
        } type;
    }

    namespace types
    {
        namespace assertions
        {
            /**
             * @brief Fundamental assertion to validate the type of either a floating point or integral type
             **/
            template <typename T>
            __device__ __host__ inline consteval void validate() noexcept
            {
                static_assert((std::is_floating_point_v<T>) || (std::is_integral_v<T>), "T must be either floating point or integral");

                if constexpr (std::is_floating_point_v<T>)
                {
                    static_assert(((std::is_same_v<T, float>) || (std::is_same_v<T, double>)), "Unsupported SCALAR_PRECISION value (must be 32 or 64)");
                }
                if constexpr (std::is_integral_v<T>)
                {
                    static_assert(((std::is_same_v<T, uint32_t>) || (std::is_same_v<T, host::label_t>)), "Unsupported LABEL_SIZE value (must be 32 or 64)");
                }
            }
        }
    }
}

#include "axisTypedefs.cuh"
#include "velocityTypedefs.cuh"
#include "ptrTypedefs.cuh"
#include "var3.cuh"
#include "coordinateTypedefs.cuh"

namespace LBM
{
    namespace axis
    {
        template <const axis::type alpha, const int coeff>
        __host__ [[nodiscard]] static inline constexpr host::blockLabel to_3d(const host::label_t ta, const host::label_t tb) noexcept
        {
            if constexpr (alpha == axis::X)
            {
                return {thread::boundary<alpha, coeff>(), ta, tb};
            }

            if constexpr (alpha == axis::Y)
            {
                return {ta, thread::boundary<alpha, coeff>(), tb};
            }

            if constexpr (alpha == axis::Z)
            {
                return {ta, tb, thread::boundary<alpha, coeff>()};
            }
        }
    }
}

#endif
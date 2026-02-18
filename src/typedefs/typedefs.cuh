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
     * @brief Shorthand for std::string and std::vector<std::string>
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
    }

    /**
     * @brief Enumerated variable indices
     **/
    namespace index
    {
        typedef enum Enum : std::size_t
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
}

#include "integralTypedefs.cuh"
#include "arithmeticTypedefs.cuh"
#include "axisTypedefs.cuh"
#include "velocityTypedefs.cuh"
#include "ptrTypedefs.cuh"

#include "../hardwareConfig.cuh"
#include "../errorHandler.cuh"

namespace LBM
{
    /**
     * @brief Struct used to represent 2D indices in a more readable way
     **/
    template <const axis::type alpha>
    class dim2
    {
    public:
        /**
         * @brief Constructs from a linear index of a flattened 2D array with dimensions (block::n<alpha>(), block::n<beta>())
         * @param[in] linearIdx The linear index to convert to 2D indices
         **/
        __device__ __host__ [[nodiscard]] inline constexpr dim2(const label_t linearIdx) noexcept
            : i_(linearIdx % (block::n<axis::orthogonal<alpha, 0>()>())),
              j_(linearIdx / (block::n<axis::orthogonal<alpha, 0>()>()))
        {
            axis::assertions::validate<alpha, axis::NOT_NULL>();
        };

        __device__ __host__ [[nodiscard]] inline constexpr dim2(const label_t a, const label_t b) noexcept
            : i_(a),
              j_(b)
        {
            axis::assertions::validate<alpha, axis::NOT_NULL>();
        };

        __device__ __host__ [[nodiscard]] inline constexpr label_t i() const noexcept
        {
            return i_;
        }

        __device__ __host__ [[nodiscard]] inline constexpr label_t j() const noexcept
        {
            return j_;
        }

    private:
        const label_t i_;
        const label_t j_;
    };
}

#endif
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
    A list of integral typedefs used throughout the cudaLBM source code

Namespace
    LBM

SourceFiles
    labelTypedefs.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_LABELTYPEDEFS_CUH
#define __MBLBM_LABELTYPEDEFS_CUH

namespace LBM
{
    /**
     * @brief Unsigned integral type used for label types
     * @note Types are either 32 bit or 64 bit unsigned integers
     * @note These types are supplied via command line defines during compilation
     **/
#ifdef LABEL_SIZE
#if LABEL_SIZE == 32
    typedef uint32_t label_t;
#elif LABEL_SIZE == 64
    typedef uint64_t label_t;
#else
    static_assert(false, "Unsupported LABEL_SIZE value (must be 32 or 64)");
    typedef uint64_t label_t;
#endif
#else
    static_assert(false, "LABEL_SIZE not defined");
    typedef uint64_t label_t;
#endif

    /**
     * @brief Label type used for GPU indices
     * @note Has to be int because cudaSetDevice operates on int
     **/
    typedef int deviceIndex_t;

    /**
     * @brief Label type used for MPI ranks
     * @note Has to be int because MPI_Comm_rank and etc take &int
     **/
    typedef int mpiRank_t;

    /**
     * @brief CUDA implementation of a std::integral constant
     * @param[in] T The type of integral value
     * @param[in] v The value
     **/
    template <typename T, T v>
    struct integralConstant
    {
        static constexpr const T value = v;
        using value_type = T;
        using type = integralConstant;
        __device__ __host__ [[nodiscard]] inline consteval operator value_type() const noexcept { return value; }
        __device__ __host__ [[nodiscard]] inline consteval value_type operator()() const noexcept { return value; }
    };

    /**
     * @brief Type used for compile-time indices
     **/
    template <const label_t label>
    using label_constant = const integralConstant<label_t, label>;
    template <const std::size_t label>
    using size_constant = const integralConstant<std::size_t, label>;
    template <const label_t label>
    using q_i = const integralConstant<label_t, label>;
    template <const label_t label>
    using m_i = const integralConstant<label_t, label>;

    /**
     * @brief Block dimensions descriptor
     * @details Stores lattice dimensions in 3D space
     **/
    struct blockLabel_t
    {
        const label_t nx; // < Lattice points in x-direction
        const label_t ny; // < Lattice points in y-direction
        const label_t nz; // < Lattice points in z-direction

        __device__ __host__ [[nodiscard]] inline constexpr label_t size() const noexcept
        {
            return nx * ny * nz;
        }

        void print(const name_t &name) const noexcept
        {
            std::cout << "    " << name << " = [" << nx << ", " << ny << ", " << nz << "];" << std::endl;
        }
    };
}

#endif
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
    ptrTypedefs.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_PTRTYPEDEFS_CUH
#define __MBLBM_PTRTYPEDEFS_CUH

namespace LBM
{
    namespace device
    {
        /**
         * @brief Class holding N device pointers of type T
         **/
        template <const label_t N, typename T>
        class ptrCollection
        {
        public:
            static_assert(N > 0, "N must be positive"); // Ensure N is valid

            /**
             * @brief Variadic constructor: construct from an arbitrary number of pointers
             * @return A pointer collection object constructed from args
             * @param[in] args An arbitrary number N of pointers of type T
             **/
            template <typename... Args>
            __device__ __host__ constexpr ptrCollection(const Args... args)
                : ptrs_{args...} // Initialize array with arguments
            {
                static_assert(sizeof...(Args) == N, "Incorrect number of arguments");

                static_assert((std::is_convertible_v<Args, T *> && ...), "All arguments must be convertible to T*");
            }

            /**
             * @brief Provides access to the GPU pointer
             * @param[in] i The index of the pointer
             **/
            template <const label_t i>
            __device__ __host__ [[nodiscard]] inline constexpr T *ptr() const noexcept
            {
                static_assert(i < N, "Invalid pointer access");

                return ptrs_[i];
            }

            /**
             * @brief Element access operator
             * @param[in] i Index of element to access
             * @return Value at index @p i
             * @warning No bounds checking performed
             **/
            __device__ __host__ [[nodiscard]] inline const T *operator[](const label_t i) const noexcept
            {
                return ptrs_[i];
            }

        private:
            /**
             * @brief The underlying pointers
             **/
            T *const ptrRestrict ptrs_[N];
        };
    }
}

#endif
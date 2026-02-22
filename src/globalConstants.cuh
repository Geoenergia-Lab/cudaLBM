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
    globalConstants.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_GLOBALCONSTANTS_CUH
#define __MBLBM_GLOBALCONSTANTS_CUH

#include "errorHandler.cuh"

namespace LBM
{
    /**
     * @brief Number of hydrodynamic moments
     **/
    template <typename T = label_t>
    __device__ __host__ [[nodiscard]] inline consteval T NUMBER_MOMENTS() noexcept
    {
        return 10;
    }

    /**
     * @brief Reference density 1.0
     **/
    template <typename T = scalar_t>
    __device__ __host__ [[nodiscard]] inline consteval T rho0() noexcept
    {
        return static_cast<T>(1);
    }

    namespace device
    {
        /**
         * @brief Characteristic physical variables
         **/
        __device__ __constant__ scalar_t Re;
        __device__ __constant__ scalar_t tau;
        __device__ __constant__ scalar_t L_char;
        __device__ __constant__ scalar_t omega;
        __device__ __constant__ scalar_t t_omegaVar;
        __device__ __constant__ scalar_t omegaVar_d2;

        /**
         * @brief Boundary condition variables
         **/
        __device__ __constant__ scalar_t U_North[3];
        __device__ __constant__ scalar_t U_South[3];
        __device__ __constant__ scalar_t U_East[3];
        __device__ __constant__ scalar_t U_West[3];
        __device__ __constant__ scalar_t U_Back[3];
        __device__ __constant__ scalar_t U_Front[3];

        /**
         * @brief Mesh constant variables
         **/
        __device__ __constant__ label_t nx;
        __device__ __constant__ label_t ny;
        __device__ __constant__ label_t nz;
        __device__ __constant__ label_t NUM_BLOCK_X;
        __device__ __constant__ label_t NUM_BLOCK_Y;
        __device__ __constant__ label_t NUM_BLOCK_Z;
        __device__ __constant__ label_t BLOCK_OFFSET_X;
        __device__ __constant__ label_t BLOCK_OFFSET_Y;
        __device__ __constant__ label_t BLOCK_OFFSET_Z;

        /**
         * @brief Allocates a symbol of type T to the device
         * @param[in] symbol The symbol to which the value is to be copied
         * @param[in] value The value to copy to the symbol
         **/
        template <typename T>
        void copyToSymbol(const T &symbol, const T value)
        {
            errorHandler::check(cudaDeviceSynchronize());
            const T valueTemp = value;
            errorHandler::check(cudaMemcpyToSymbol(symbol, &valueTemp, sizeof(T), 0, cudaMemcpyHostToDevice));
            errorHandler::check(cudaDeviceSynchronize());
        }

        template <typename T, const std::size_t N>
        void copyToSymbol(const T (&symbol)[N], const T (&value)[N])
        {
            errorHandler::check(cudaDeviceSynchronize());
            errorHandler::check(cudaMemcpyToSymbol(symbol, value, N * sizeof(T), 0, cudaMemcpyHostToDevice));
            errorHandler::check(cudaDeviceSynchronize());
        }

        template <typename T, const std::size_t N>
        void copyToSymbol(const T (&symbol)[N], const T value, const label_t index)
        {
            if (index >= N)
            {
                throw std::runtime_error("Error setting device symbol index" + std::to_string(index) + " out of bounds for array of size " + std::to_string(N) + ".");
            }
            errorHandler::check(cudaDeviceSynchronize());
            const T valueTemp = value;
            errorHandler::check(cudaMemcpyToSymbol(symbol, &valueTemp, sizeof(T), static_cast<std::size_t>(index) * sizeof(T), cudaMemcpyHostToDevice));
            errorHandler::check(cudaDeviceSynchronize());
        }
    }
}

#endif
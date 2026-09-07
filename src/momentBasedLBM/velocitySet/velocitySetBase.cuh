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
    Common methods shared by all velocity sets

Namespace
    LBM

SourceFiles
    velocitySetBase.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_VELOCITYSETBASE_CUH
#define __MBLBM_VELOCITYSETBASE_CUH

namespace LBM
{
    class velocitySetBase
    {
    public:
        /**
         * @brief Get the a^2 constant (3.0)
         * @tparam T The return type
         **/
        template <typename T>
        __device__ __host__ [[nodiscard]] static inline consteval T as2() noexcept
        {
            return static_cast<T>(3);
        }

        /**
         * @brief Get the speed of sound squared (c^2 = 1 / 3)
         * @tparam T The return type
         **/
        template <typename T>
        __device__ __host__ [[nodiscard]] static inline consteval T cs2() noexcept
        {
            return static_cast<T>(static_cast<double>(1) / static_cast<double>(3));
        }

        /**
         * @brief Get scaling factor for first-order moments
         * @tparam T The return type
         **/
        template <typename T>
        __device__ __host__ [[nodiscard]] static inline consteval T scale_i() noexcept
        {
            return static_cast<T>(3);
        }

        /**
         * @brief Get scaling factor for diagonal second-order moments
         * @tparam T The return type
         **/
        template <typename T>
        __device__ __host__ [[nodiscard]] static inline consteval T scale_ii() noexcept
        {
            return static_cast<T>(4.5);
        }

        /**
         * @brief Get scaling factor for off-diagonal second-order moments
         * @tparam T The return type
         **/
        template <typename T>
        __device__ __host__ [[nodiscard]] static inline consteval T scale_ij() noexcept
        {
            return static_cast<T>(9);
        }

        /**
         * @brief Get scaling factor for potentially diagonal or off-diagonal second-order moments
         * @tparam T The return type
         * @param[in] is_diagonal Boolean indicating whether the moment is diagonal (true) or off-diagonal (false)
         * @return Scaling factor for the second-order moment
         **/
        template <typename T>
        __device__ __host__ [[nodiscard]] static inline constexpr T scale(const bool is_diagonal) noexcept
        {
            if (is_diagonal)
            {
                return static_cast<T>(4.5);
            }
            else
            {
                return static_cast<T>(9);
            }
        }

        /**
         * @brief Apply velocity set scaling factors to moment array
         * @param[out] moments Moment array (rho, U, Pi)
         *
         * This method applies the appropriate scaling factors to each moment component:
         * - First-order moments (velocity components): scaled by scale_i()
         * - Diagonal second-order moments: scaled by scale_ii()
         * - Off-diagonal second-order moments: scaled by scale_ij()
         **/
        __device__ __host__ static inline void scale(momentsArray &moments) noexcept
        {
            // Scale the moments correctly
            moments[m_i<1>()] = scale_i<scalar_t>() * (moments[m_i<1>()]);
            moments[m_i<2>()] = scale_i<scalar_t>() * (moments[m_i<2>()]);
            moments[m_i<3>()] = scale_i<scalar_t>() * (moments[m_i<3>()]);
            moments[m_i<4>()] = scale_ii<scalar_t>() * (moments[m_i<4>()]);
            moments[m_i<5>()] = scale_ij<scalar_t>() * (moments[m_i<5>()]);
            moments[m_i<6>()] = scale_ij<scalar_t>() * (moments[m_i<6>()]);
            moments[m_i<7>()] = scale_ii<scalar_t>() * (moments[m_i<7>()]);
            moments[m_i<8>()] = scale_ij<scalar_t>() * (moments[m_i<8>()]);
            moments[m_i<9>()] = scale_ii<scalar_t>() * (moments[m_i<9>()]);
        }
    };
}

#endif

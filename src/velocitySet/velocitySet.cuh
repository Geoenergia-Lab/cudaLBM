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
    Top-level header file for the velocity set classes

Namespace
    LBM

SourceFiles
    D3Q19.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_VELOCITYSET_CUH
#define __MBLBM_VELOCITYSET_CUH

#include "../LBMIncludes.cuh"
#include "../LBMTypedefs.cuh"
#include "../globalFunctions.cuh"
#include "../array/threadArray.cuh"

namespace LBM
{
    /**
     * @class velocitySet
     * @brief Base class for LBM velocity sets providing common constants and scaling operations
     *
     * This class serves as a base for specific velocity set implementations (e.g., D3Q19, D3Q27)
     * and provides common constants, scaling factors, and utility functions used across
     * different velocity set configurations in the Lattice Boltzmann Method.
     **/
    class velocitySet
    {
    public:
        /**
         * @brief Default constructor (consteval)
         **/
        __device__ __host__ [[nodiscard]] inline consteval velocitySet() noexcept {};

        /**
         * @brief Get the speed of sound (sqrt(3))
         **/
        template <typename T>
        __device__ __host__ [[nodiscard]] static inline consteval T as() noexcept
        {
            return static_cast<T>(std::sqrt(3));
        }

        /**
         * @brief Get the a^2 constant (a ^ 2)
         **/
        template <typename T>
        __device__ __host__ [[nodiscard]] static inline consteval T as2() noexcept
        {
            return static_cast<T>(3);
        }

        /**
         * @brief Get the speed of sound squared (c^2 = 1 / 3)
         **/
        template <typename T>
        __device__ __host__ [[nodiscard]] static inline consteval T cs2() noexcept
        {
            return static_cast<T>(static_cast<double>(1) / static_cast<double>(3));
        }

        /**
         * @brief Get scaling factor for first-order moments
         **/
        template <typename T>
        __device__ __host__ [[nodiscard]] static inline consteval T scale_i() noexcept
        {
            return static_cast<T>(3);
        }

        /**
         * @brief Get scaling factor for diagonal second-order moments
         **/
        template <typename T>
        __device__ __host__ [[nodiscard]] static inline consteval T scale_ii() noexcept
        {
            return static_cast<T>(4.5);
        }

        /**
         * @brief Get scaling factor for off-diagonal second-order moments
         **/
        template <typename T>
        __device__ __host__ [[nodiscard]] static inline consteval T scale_ij() noexcept
        {
            return static_cast<T>(9);
        }

        /**
         * @brief Apply velocity set scaling factors to moment array
         * @param[in,out] moments Array of 10 moment variables to be scaled
         *
         * This method applies the appropriate scaling factors to each moment component:
         * - First-order moments (velocity components): scaled by scale_i()
         * - Diagonal second-order moments: scaled by scale_ii()
         * - Off-diagonal second-order moments: scaled by scale_ij()
         **/
        __device__ static inline void scale(thread::array<scalar_t, 10> &moments) noexcept
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

        template <class VelocitySet, const axisDirection dir, const label_t q_>
        __device__ __host__ [[nodiscard]] static inline consteval bool is_negative(const q_i<q_> q) noexcept
        {
            static_assert(((dir == X) || (dir == Y) || (dir == Z)));

            if constexpr (dir == X)
            {
                return (VelocitySet::template cx<int>(q) < 0);
            }

            if constexpr (dir == Y)
            {
                return (VelocitySet::template cy<int>(q) < 0);
            }

            if constexpr (dir == Z)
            {
                return (VelocitySet::template cz<int>(q) < 0);
            }
        }

        template <class VelocitySet, const axisDirection dir, const label_t q_>
        __device__ __host__ [[nodiscard]] static inline consteval bool is_positive(const q_i<q_> q) noexcept
        {
            static_assert(((dir == X) || (dir == Y) || (dir == Z)));

            if constexpr (dir == X)
            {
                return (VelocitySet::template cx<int>(q) > 0);
            }

            if constexpr (dir == Y)
            {
                return (VelocitySet::template cy<int>(q) > 0);
            }

            if constexpr (dir == Z)
            {
                return (VelocitySet::template cz<int>(q) > 0);
            }
        }

        /**
         * @brief Determines if a discrete velocity direction is incoming relative to a boundary normal
         * @tparam T Return type (typically numeric type)
         * @tparam BoundaryNormal Type of boundary normal object with directional methods
         * @tparam q_ Compile-time velocity direction index
         * @param[in] q Compile-time constant representing velocity direction
         * @param[in] boundaryNormal Boundary normal information with directional methods
         * @return T 1 if velocity is incoming (pointing into domain), 0 if outgoing
         *
         * @details Checks if velocity components oppose boundary normal direction:
         * - For East boundary (normal.x > 0): checks negative x-velocity component
         * - For West boundary (normal.x < 0): checks positive x-velocity component
         * - For North boundary (normal.y > 0): checks negative y-velocity component
         * - For South boundary (normal.y < 0): checks positive y-velocity component
         * - For Front boundary (normal.z > 0): checks negative z-velocity component
         * - For Back boundary (normal.z < 0): checks positive z-velocity component
         * Returns 1 only if no incoming component is detected on any axis
         **/
        template <typename T, class VelocitySet, class BoundaryNormal, const label_t q_>
        __device__ __host__ [[nodiscard]] static inline constexpr T incomingSwitch(const q_i<q_> q, const BoundaryNormal &boundaryNormal) noexcept
        {
            // boundaryNormal.x > 0  => EAST boundary
            // boundaryNormal.x < 0  => WEST boundary
            const bool cond_x = (boundaryNormal.isEast() & is_negative<VelocitySet, X>(q)) | (boundaryNormal.isWest() & is_positive<VelocitySet, X>(q));

            // boundaryNormal.y > 0  => NORTH boundary
            // boundaryNormal.y < 0  => SOUTH boundary
            const bool cond_y = (boundaryNormal.isNorth() & is_negative<VelocitySet, Y>(q)) | (boundaryNormal.isSouth() & is_positive<VelocitySet, Y>(q));

            // boundaryNormal.z > 0  => FRONT boundary
            // boundaryNormal.z < 0  => BACK boundary
            const bool cond_z = (boundaryNormal.isFront() & is_negative<VelocitySet, Z>(q)) | (boundaryNormal.isBack() & is_positive<VelocitySet, Z>(q));

            return static_cast<T>(!(cond_x | cond_y | cond_z));
        }

        /**
         * @brief Calculate equilibrium distribution function for a direction
         * @tparam T Data type for calculation
         * @param[in] rhow Weighted density (w_q[q] * rho)
         * @param[in] uc3 3 * (u·c_q) = 3*(u*cx + v*cy + w*cz)
         * @param[in] p1_muu 1 - 1.5*(u² + v² + w²)
         * @return Equilibrium distribution value for the direction
         **/
        template <typename T>
        __host__ [[nodiscard]] static T f_eq(const T rhow, const T uc3, const T p1_muu) noexcept
        {
            return (rhow * (p1_muu + uc3 * (static_cast<T>(1.0) + uc3 * static_cast<T>(0.5))));
        }

        /**
         * @brief Calculate full equilibrium distribution for given velocity
         * @tparam T Data type for calculation
         * @param[in] u x-component of velocity
         * @param[in] v y-component of velocity
         * @param[in] w z-component of velocity
         * @return Array of 19 equilibrium distribution values
         **/
        template <class VelocitySet>
        __host__ [[nodiscard]] static const thread::array<scalar_t, VelocitySet::Q()> F_eq(const scalar_t u, const scalar_t v, const scalar_t w) noexcept
        {
            thread::array<scalar_t, VelocitySet::Q()> pop;

            for (label_t q = 0; q < VelocitySet::Q(); q++)
            {
                pop[q] = f_eq<scalar_t>(
                    VelocitySet::template w_q<scalar_t>()[q],
                    static_cast<scalar_t>(3) * ((u * VelocitySet::template cx<scalar_t>()[q]) + (v * VelocitySet::template cy<scalar_t>()[q]) + (w * VelocitySet::template cz<scalar_t>()[q])),
                    static_cast<scalar_t>(1) - static_cast<scalar_t>(1.5) * ((u * u) + (v * v) + (w * w)));
            }

            return pop;
        }

        template <class VelocitySet, const axisDirection alpha, const axisDirection beta>
        __device__ __host__ [[nodiscard]] static inline consteval const thread::array<int, VelocitySet::Q()> c_AlphaBeta() noexcept
        {
            static_assert(((alpha == X) || (alpha == Y) || (alpha == Z) || (alpha == NO_DIRECTION)));
            static_assert(((beta == X) || (beta == Y) || (beta == Z) || (beta == NO_DIRECTION)));

            if constexpr ((alpha == NO_DIRECTION) && (beta == NO_DIRECTION))
            {
                thread::array<int, VelocitySet::Q()> toReturn;
                for (std::size_t i = 0; i < toReturn.size(); i++)
                {
                    toReturn[i] = 1;
                }
                return toReturn;
            }

            if constexpr ((alpha == X))
            {
                if constexpr ((beta == X))
                {
                    return (VelocitySet::template cx<int>() * VelocitySet::template cx<int>());
                }
                if constexpr ((beta == Y))
                {
                    return (VelocitySet::template cx<int>() * VelocitySet::template cy<int>());
                }
                if constexpr ((beta == Z))
                {
                    return (VelocitySet::template cx<int>() * VelocitySet::template cz<int>());
                }
                if constexpr ((beta == NO_DIRECTION))
                {
                    return VelocitySet::template cx<int>();
                }
            }

            if constexpr ((alpha == Y))
            {
                if constexpr ((beta == X))
                {
                    return (VelocitySet::template cy<int>() * VelocitySet::template cx<int>());
                }
                if constexpr ((beta == Y))
                {
                    return (VelocitySet::template cy<int>() * VelocitySet::template cy<int>());
                }
                if constexpr ((beta == Z))
                {
                    return (VelocitySet::template cy<int>() * VelocitySet::template cz<int>());
                }
                if constexpr ((beta == NO_DIRECTION))
                {
                    return VelocitySet::template cy<int>();
                }
            }

            if constexpr ((alpha == Z))
            {
                if constexpr ((beta == X))
                {
                    return (VelocitySet::template cz<int>() * VelocitySet::template cx<int>());
                }
                if constexpr ((beta == Y))
                {
                    return (VelocitySet::template cz<int>() * VelocitySet::template cy<int>());
                }
                if constexpr ((beta == Z))
                {
                    return (VelocitySet::template cz<int>() * VelocitySet::template cz<int>());
                }
                if constexpr ((beta == NO_DIRECTION))
                {
                    return VelocitySet::template cz<int>();
                }
            }
        }

        template <const int coeff, class VelocitySet, const label_t I, class BoundaryNormal>
        __device__ __host__ [[nodiscard]] static inline constexpr scalar_t process_momentum_element(
            const scalar_t pop_value,
            const BoundaryNormal &boundaryNormal) noexcept
        {
            static_assert(((coeff == -1) || (coeff == 1)), "Invalid coefficient");

            if constexpr (coeff == 1)
            {
                return incomingSwitch<scalar_t, VelocitySet>(q_i<I>(), boundaryNormal) * pop_value;
            }
            else if constexpr (coeff == -1)
            {
                return -incomingSwitch<scalar_t, VelocitySet>(q_i<I>(), boundaryNormal) * pop_value;
            }
        }

        template <const int coeff, class VelocitySet>
        __device__ __host__ [[nodiscard]] static inline constexpr scalar_t process_momentum_element(
            const scalar_t pop_value) noexcept
        {
            static_assert(((coeff == -1) || (coeff == 1)), "Invalid coefficient");

            if constexpr (coeff == 1)
            {
                return pop_value;
            }
            else if constexpr (coeff == -1)
            {
                return -pop_value;
            }
        }

        template <class VelocitySet, const axisDirection alpha, const axisDirection beta, class BoundaryNormal>
        __device__ __host__ [[nodiscard]] static inline constexpr scalar_t calculate_moment(const thread::array<scalar_t, VelocitySet::Q()> &pop, const BoundaryNormal &boundaryNormal) noexcept
        {
            constexpr const thread::array<int, VelocitySet::Q()> c_AB = c_AlphaBeta<VelocitySet, alpha, beta>();
            constexpr const label_t N = number_non_zero(c_AB);
            constexpr const thread::array<int, N> C = non_zero_values<N>(c_AB);
            constexpr const thread::array<label_t, N> indices = non_zero_indices<N>(c_AB);

            return [&]<const label_t... Is>(std::index_sequence<Is...>)
            {
                return (process_momentum_element<C[Is], VelocitySet, indices[Is]>(pop[indices[Is]], boundaryNormal) + ...);
            }(std::make_index_sequence<N>{});
        }

        template <class VelocitySet, const axisDirection alpha, const axisDirection beta>
        __device__ __host__ [[nodiscard]] static inline constexpr scalar_t calculate_moment(const thread::array<scalar_t, VelocitySet::Q()> &pop) noexcept
        {
            constexpr const thread::array<int, VelocitySet::Q()> c_AB = c_AlphaBeta<VelocitySet, alpha, beta>();
            constexpr const label_t N = number_non_zero(c_AB);
            constexpr const thread::array<int, N> C = non_zero_values<N>(c_AB);
            constexpr const thread::array<label_t, N> indices = non_zero_indices<N>(c_AB);

            return [&]<const label_t... Is>(std::index_sequence<Is...>)
            {
                return (process_momentum_element<C[Is], VelocitySet>(pop[indices[Is]]) + ...);
            }(std::make_index_sequence<N>{});
        }

        template <class VelocitySet>
        __device__ __host__ static inline void calculate_moments(const thread::array<scalar_t, VelocitySet::Q()> &pop, thread::array<scalar_t, NUMBER_MOMENTS()> &mom) noexcept
        {
            // Density
            mom[m_i<0>()] = calculate_moment<VelocitySet, NO_DIRECTION, NO_DIRECTION>(pop);
            const scalar_t inv_rho = static_cast<scalar_t>(1) / mom[m_i<0>()];

            // Velocity
            mom[m_i<1>()] = calculate_moment<VelocitySet, X, NO_DIRECTION>(pop) * inv_rho;
            mom[m_i<2>()] = calculate_moment<VelocitySet, Y, NO_DIRECTION>(pop) * inv_rho;
            mom[m_i<3>()] = calculate_moment<VelocitySet, Z, NO_DIRECTION>(pop) * inv_rho;

            // Second order moments
            mom[m_i<4>()] = (calculate_moment<VelocitySet, X, X>(pop) * inv_rho) - cs2<scalar_t>();
            mom[m_i<5>()] = calculate_moment<VelocitySet, X, Y>(pop) * inv_rho;
            mom[m_i<6>()] = calculate_moment<VelocitySet, X, Z>(pop) * inv_rho;
            mom[m_i<7>()] = (calculate_moment<VelocitySet, Y, Y>(pop) * inv_rho) - cs2<scalar_t>();
            mom[m_i<8>()] = calculate_moment<VelocitySet, Y, Z>(pop) * inv_rho;
            mom[m_i<9>()] = (calculate_moment<VelocitySet, Z, Z>(pop) * inv_rho) - cs2<scalar_t>();
        }

        template <class VelocitySet, class BoundaryNormal>
        __device__ __host__ static inline void calculate_moments(const thread::array<scalar_t, VelocitySet::Q()> &pop, thread::array<scalar_t, NUMBER_MOMENTS()> &mom, const BoundaryNormal &boundaryNormal) noexcept
        {
            // Density
            mom[m_i<0>()] = calculate_moment<VelocitySet, NO_DIRECTION, NO_DIRECTION>(pop, boundaryNormal);
            const scalar_t inv_rho = static_cast<scalar_t>(1) / mom[m_i<0>()];

            // Velocity
            mom[m_i<1>()] = calculate_moment<VelocitySet, X, NO_DIRECTION>(pop, boundaryNormal) * inv_rho;
            mom[m_i<2>()] = calculate_moment<VelocitySet, Y, NO_DIRECTION>(pop, boundaryNormal) * inv_rho;
            mom[m_i<3>()] = calculate_moment<VelocitySet, Z, NO_DIRECTION>(pop, boundaryNormal) * inv_rho;

            // Second order moments
            mom[m_i<4>()] = (calculate_moment<VelocitySet, X, X>(pop, boundaryNormal) * inv_rho) - cs2<scalar_t>();
            mom[m_i<5>()] = calculate_moment<VelocitySet, X, Y>(pop, boundaryNormal) * inv_rho;
            mom[m_i<6>()] = calculate_moment<VelocitySet, X, Z>(pop, boundaryNormal) * inv_rho;
            mom[m_i<7>()] = (calculate_moment<VelocitySet, Y, Y>(pop, boundaryNormal) * inv_rho) - cs2<scalar_t>();
            mom[m_i<8>()] = calculate_moment<VelocitySet, Y, Z>(pop, boundaryNormal) * inv_rho;
            mom[m_i<9>()] = (calculate_moment<VelocitySet, Z, Z>(pop, boundaryNormal) * inv_rho) - cs2<scalar_t>();
        }
    };
}

#include "D3Q19.cuh"
#include "D3Q27.cuh"

#endif
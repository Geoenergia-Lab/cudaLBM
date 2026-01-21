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
    Definition of the D3Q19 velocity set

Namespace
    LBM

SourceFiles
    D3Q19.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_D3Q19_CUH
#define __MBLBM_D3Q19_CUH

#include "velocitySet.cuh"

namespace LBM
{
    /**
     * @class D3Q19
     * @brief Implements the D3Q19 velocity set for 3D Lattice Boltzmann simulations
     * @extends velocitySet
     *
     * This class provides the specific implementation for the D3Q19 lattice model,
     * which includes 19 discrete velocity directions in 3D space. It contains:
     * - Velocity components (cx, cy, cz) for each direction
     * - Weight coefficients for each direction
     * - Methods for moment calculation and population reconstruction
     * - Equilibrium distribution functions
     **/
    class D3Q19 : private velocitySet
    {
    public:
        /**
         * @brief Default constructor (consteval)
         **/
        __device__ __host__ [[nodiscard]] inline consteval D3Q19(){};

        /**
         * @brief Get number of discrete velocity directions
         * @return 19 (number of directions in D3Q19 lattice)
         **/
        __device__ __host__ [[nodiscard]] static inline consteval label_t Q() noexcept
        {
            return Q_;
        }

        /**
         * @brief Get number of velocity components on a lattice face
         * @return 5 (number of directions crossing each face in D3Q19)
         **/
        __device__ __host__ [[nodiscard]] static inline consteval label_t QF() noexcept
        {
            return QF_;
        }

        /**
         * @brief Get weight for stationary component (q=0)
         **/
        template <typename T>
        __device__ __host__ [[nodiscard]] static inline consteval T w_0() noexcept
        {
            return static_cast<T>(static_cast<double>(1) / static_cast<double>(3));
        }

        /**
         * @brief Get weight for orthogonal directions (q=1-6)
         **/
        template <typename T>
        __device__ __host__ [[nodiscard]] static inline consteval T w_1() noexcept
        {
            return static_cast<T>(static_cast<double>(1) / static_cast<double>(18));
        }

        /**
         * @brief Get weight for diagonal directions (q=7-18)
         **/
        template <typename T>
        __device__ __host__ [[nodiscard]] static inline consteval T w_2() noexcept
        {
            return static_cast<T>(static_cast<double>(1) / static_cast<double>(36));
        }

        /**
         * @brief Get all weights for computation
         * @return Thread array of 19 weights in D3Q19 order
         **/
        template <typename T>
        __device__ __host__ [[nodiscard]] static inline consteval const thread::array<T, 19> w_q() noexcept
        {
            return {
                w_0<T>(),
                w_1<T>(), w_1<T>(), w_1<T>(), w_1<T>(), w_1<T>(), w_1<T>(),
                w_2<T>(), w_2<T>(), w_2<T>(), w_2<T>(), w_2<T>(), w_2<T>(), w_2<T>(), w_2<T>(), w_2<T>(), w_2<T>(), w_2<T>(), w_2<T>()};
        }

        /**
         * @brief Get weight for specific direction
         * @tparam q_ Direction index (0-18)
         * @param[in] q Direction index as compile-time constant
         * @return Weight for specified direction
         **/
        template <typename T, const label_t q_>
        __device__ __host__ [[nodiscard]] static inline consteval T w_q(const q_i<q_> q) noexcept
        {
            // Check that we are accessing a valid member
            static_assert(q() < Q_, "Invalid velocity set index in member function w(q)");

            // Return the component
            return w_q<T>()[q()];
        }

        /**
         * @brief Get x-components for all directions
         * @return Thread array of 19 x-velocity components
         **/
        template <typename T>
        __device__ __host__ [[nodiscard]] static inline consteval const thread::array<T, 19> cx() noexcept
        {
            return {static_cast<T>(0), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(0), static_cast<T>(0)};
        }

        /**
         * @brief Get x-component for specific direction
         * @tparam q_ Direction index (0-18)
         * @param[in] q Direction index as compile-time constant
         * @return x-component for specified direction
         **/
        template <typename T, const label_t q_>
        __device__ __host__ [[nodiscard]] static inline consteval T cx(const q_i<q_> q) noexcept
        {
            // Check that we are accessing a valid member
            static_assert(q() < Q_, "Invalid velocity set index in member function cx(q)");

            // Return the component
            return cx<T>()[q()];
        }

        /**
         * @brief Get y-components for all directions
         * @return Thread array of 19 y-velocity components
         **/
        template <typename T>
        __device__ __host__ [[nodiscard]] static inline consteval const thread::array<T, 19> cy() noexcept
        {
            return {static_cast<T>(0), static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(-1), static_cast<T>(1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(-1)};
        }

        /**
         * @brief Get y-component for specific direction
         * @tparam q_ Direction index (0-18)
         * @param[in] q Direction index as compile-time constant
         * @return y-component for specified direction
         **/
        template <typename T, const label_t q_>
        __device__ __host__ [[nodiscard]] static inline consteval T cy(const q_i<q_> q) noexcept
        {
            // Check that we are accessing a valid member
            static_assert(q() < Q_, "Invalid velocity set index in member function cy(q)");

            // Return the component
            return cy<T>()[q()];
        }

        /**
         * @brief Get z-components for all directions
         * @return Thread array of 19 z-velocity components
         **/
        template <typename T>
        __device__ __host__ [[nodiscard]] static inline consteval const thread::array<T, 19> cz() noexcept
        {
            return {static_cast<T>(0), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(-1), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(1)};
        }

        /**
         * @brief Get z-component for specific direction
         * @tparam q_ Direction index (0-18)
         * @param[in] q Direction index as compile-time constant
         * @return z-component for specified direction
         **/
        template <typename T, const label_t q_>
        __device__ __host__ [[nodiscard]] static inline consteval T cz(const q_i<q_> q) noexcept
        {
            // Check that we are accessing a valid member
            static_assert(q() < Q_, "Invalid velocity set index in member function cz(q)");

            // Return the component
            return cz<T>()[q()];
        }

        /**
         * @brief Get alpha-components for all directions
         * @return Thread array of 19 alpha-velocity components
         **/
        template <typename T, const axisDirection alpha>
        __device__ __host__ [[nodiscard]] static inline consteval const thread::array<T, 19> c() noexcept
        {
            if constexpr (alpha == axisDirection::NO_DIRECTION)
            {
                thread::array<T, Q_> result;
                for (std::size_t i = 0; i < Q_; i++)
                {
                    result[i] = 1;
                }
                return result;
            }
            if constexpr (alpha == axisDirection::X)
            {
                return cx<T>();
            }
            if constexpr (alpha == axisDirection::Y)
            {
                return cy<T>();
            }
            if constexpr (alpha == axisDirection::Z)
            {
                return cz<T>();
            }
        }

        /**
         * @brief Reconstruct population distribution from moments (in-place)
         * @param[out] pop Population array to be filled
         * @param[in] moments Moment array (10 components)
         **/
        __device__ static inline void reconstruct(thread::array<scalar_t, 19> &pop, const thread::array<scalar_t, NUMBER_MOMENTS()> &moments) noexcept
        {
            const scalar_t pics2 = static_cast<scalar_t>(1.0) - cs2<scalar_t>() * (moments[m_i<4>()] + moments[m_i<7>()] + moments[m_i<9>()]);

            const scalar_t rhow_0 = moments[m_i<0>()] * w_0<scalar_t>();
            pop[q_i<0>()] = rhow_0 * pics2;

            const scalar_t rhow_1 = moments[m_i<0>()] * w_1<scalar_t>();
            pop[q_i<1>()] = rhow_1 * (pics2 + moments[m_i<1>()] + moments[m_i<4>()]);
            pop[q_i<2>()] = rhow_1 * (pics2 - moments[m_i<1>()] + moments[m_i<4>()]);
            pop[q_i<3>()] = rhow_1 * (pics2 + moments[m_i<2>()] + moments[m_i<7>()]);
            pop[q_i<4>()] = rhow_1 * (pics2 - moments[m_i<2>()] + moments[m_i<7>()]);
            pop[q_i<5>()] = rhow_1 * (pics2 + moments[m_i<3>()] + moments[m_i<9>()]);
            pop[q_i<6>()] = rhow_1 * (pics2 - moments[m_i<3>()] + moments[m_i<9>()]);

            const scalar_t rhow_2 = moments[m_i<0>()] * w_2<scalar_t>();
            pop[q_i<7>()] = rhow_2 * (pics2 + moments[m_i<1>()] + moments[m_i<2>()] + moments[m_i<4>()] + moments[m_i<7>()] + moments[m_i<5>()]);
            pop[q_i<8>()] = rhow_2 * (pics2 - moments[m_i<1>()] - moments[m_i<2>()] + moments[m_i<4>()] + moments[m_i<7>()] + moments[m_i<5>()]);
            pop[q_i<9>()] = rhow_2 * (pics2 + moments[m_i<1>()] + moments[m_i<3>()] + moments[m_i<4>()] + moments[m_i<9>()] + moments[m_i<6>()]);
            pop[q_i<10>()] = rhow_2 * (pics2 - moments[m_i<1>()] - moments[m_i<3>()] + moments[m_i<4>()] + moments[m_i<9>()] + moments[m_i<6>()]);
            pop[q_i<11>()] = rhow_2 * (pics2 + moments[m_i<2>()] + moments[m_i<3>()] + moments[m_i<7>()] + moments[m_i<9>()] + moments[m_i<8>()]);
            pop[q_i<12>()] = rhow_2 * (pics2 - moments[m_i<2>()] - moments[m_i<3>()] + moments[m_i<7>()] + moments[m_i<9>()] + moments[m_i<8>()]);
            pop[q_i<13>()] = rhow_2 * (pics2 + moments[m_i<1>()] - moments[m_i<2>()] + moments[m_i<4>()] + moments[m_i<7>()] - moments[m_i<5>()]);
            pop[q_i<14>()] = rhow_2 * (pics2 - moments[m_i<1>()] + moments[m_i<2>()] + moments[m_i<4>()] + moments[m_i<7>()] - moments[m_i<5>()]);
            pop[q_i<15>()] = rhow_2 * (pics2 + moments[m_i<1>()] - moments[m_i<3>()] + moments[m_i<4>()] + moments[m_i<9>()] - moments[m_i<6>()]);
            pop[q_i<16>()] = rhow_2 * (pics2 - moments[m_i<1>()] + moments[m_i<3>()] + moments[m_i<4>()] + moments[m_i<9>()] - moments[m_i<6>()]);
            pop[q_i<17>()] = rhow_2 * (pics2 + moments[m_i<2>()] - moments[m_i<3>()] + moments[m_i<7>()] + moments[m_i<9>()] - moments[m_i<8>()]);
            pop[q_i<18>()] = rhow_2 * (pics2 - moments[m_i<2>()] + moments[m_i<3>()] + moments[m_i<7>()] + moments[m_i<9>()] - moments[m_i<8>()]);
        }

        /**
         * @brief Reconstruct population distribution from moments (return)
         * @param[in] moments Moment array (10 components)
         * @return Population array with 19 components
         **/
        __device__ __host__ [[nodiscard]] static inline thread::array<scalar_t, 19> reconstruct(const thread::array<scalar_t, NUMBER_MOMENTS()> &moments) noexcept
        {
            const scalar_t pics2 = static_cast<scalar_t>(1.0) - cs2<scalar_t>() * (moments[m_i<4>()] + moments[m_i<7>()] + moments[m_i<9>()]);

            const scalar_t rhow_0 = moments[m_i<0>()] * w_0<scalar_t>();
            const scalar_t rhow_1 = moments[m_i<0>()] * w_1<scalar_t>();
            const scalar_t rhow_2 = moments[m_i<0>()] * w_2<scalar_t>();

            return {
                rhow_0 * pics2,
                rhow_1 * (pics2 + moments[m_i<1>()] + moments[m_i<4>()]),
                rhow_1 * (pics2 - moments[m_i<1>()] + moments[m_i<4>()]),
                rhow_1 * (pics2 + moments[m_i<2>()] + moments[m_i<7>()]),
                rhow_1 * (pics2 - moments[m_i<2>()] + moments[m_i<7>()]),
                rhow_1 * (pics2 + moments[m_i<3>()] + moments[m_i<9>()]),
                rhow_1 * (pics2 - moments[m_i<3>()] + moments[m_i<9>()]),
                rhow_2 * (pics2 + moments[m_i<1>()] + moments[m_i<2>()] + moments[m_i<4>()] + moments[m_i<7>()] + moments[m_i<5>()]),
                rhow_2 * (pics2 - moments[m_i<1>()] - moments[m_i<2>()] + moments[m_i<4>()] + moments[m_i<7>()] + moments[m_i<5>()]),
                rhow_2 * (pics2 + moments[m_i<1>()] + moments[m_i<3>()] + moments[m_i<4>()] + moments[m_i<9>()] + moments[m_i<6>()]),
                rhow_2 * (pics2 - moments[m_i<1>()] - moments[m_i<3>()] + moments[m_i<4>()] + moments[m_i<9>()] + moments[m_i<6>()]),
                rhow_2 * (pics2 + moments[m_i<2>()] + moments[m_i<3>()] + moments[m_i<7>()] + moments[m_i<9>()] + moments[m_i<8>()]),
                rhow_2 * (pics2 - moments[m_i<2>()] - moments[m_i<3>()] + moments[m_i<7>()] + moments[m_i<9>()] + moments[m_i<8>()]),
                rhow_2 * (pics2 + moments[m_i<1>()] - moments[m_i<2>()] + moments[m_i<4>()] + moments[m_i<7>()] - moments[m_i<5>()]),
                rhow_2 * (pics2 - moments[m_i<1>()] + moments[m_i<2>()] + moments[m_i<4>()] + moments[m_i<7>()] - moments[m_i<5>()]),
                rhow_2 * (pics2 + moments[m_i<1>()] - moments[m_i<3>()] + moments[m_i<4>()] + moments[m_i<9>()] - moments[m_i<6>()]),
                rhow_2 * (pics2 - moments[m_i<1>()] + moments[m_i<3>()] + moments[m_i<4>()] + moments[m_i<9>()] - moments[m_i<6>()]),
                rhow_2 * (pics2 + moments[m_i<2>()] - moments[m_i<3>()] + moments[m_i<7>()] + moments[m_i<9>()] - moments[m_i<8>()]),
                rhow_2 * (pics2 - moments[m_i<2>()] + moments[m_i<3>()] + moments[m_i<7>()] + moments[m_i<9>()] - moments[m_i<8>()])};
        }

        /**
         * @brief Print velocity set information to terminal
         **/
        __host__ static void print() noexcept
        {
            std::cout << "D3Q19 {w, cx, cy, cz}:" << std::endl;
            std::cout << "{" << std::endl;
            printAll();
            std::cout << "};" << std::endl;
            std::cout << std::endl;
        }

    private:
        /**
         * @brief Number of velocity components in the lattice
         **/
        static constexpr const label_t Q_ = 19;

        /**
         * @brief Number of velocity components on each lattice face
         **/
        static constexpr const label_t QF_ = 5;

        /**
         * @brief Implementation of the print loop
         * @note This function effectively unrolls the loop at compile-time and checks for its bounds
         **/
        template <const label_t q_ = 0>
        __host__ static inline void printAll(const q_i<q_> q = q_i<0>()) noexcept
        {
            // Loop over the velocity set, print to terminal
            host::constexpr_for<q(), Q()>(
                [&](const auto Q)
                {
                    std::cout
                        << "    [" << q_i<Q>() << "] = {"
                        << w_q<double>()[q_i<Q>()] << ", "
                        << cx<int>()[q_i<Q>()] << ", "
                        << cy<int>()[q_i<Q>()] << ", "
                        << cz<int>()[q_i<Q>()] << "};" << std::endl;
                });
        }
    };
}

#endif
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
    Definition of the D3Q27 velocity set

Namespace
    LBM

SourceFiles
    D3Q27.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_D3Q27_CUH
#define __MBLBM_D3Q27_CUH

namespace LBM
{
    namespace constants
    {
        struct D3Q27
        {
            /**
             * @brief Get number of discrete velocity directions
             * @return 27 (number of directions in D3Q27 lattice)
             **/
            __device__ __host__ [[nodiscard]] static inline consteval label_t Q() noexcept
            {
                return 27;
            }

            /**
             * @brief Get number of velocity components on a lattice face
             * @return 9 (number of directions crossing each face in D3Q27)
             **/
            __device__ __host__ [[nodiscard]] static inline consteval label_t QF() noexcept
            {
                return 9;
            }
        };
    }

    /**
     * @class D3Q27
     * @brief Implements the D3Q27 velocity set for 3D Lattice Boltzmann simulations
     * @extends velocitySet
     *
     * This class provides the specific implementation for the D3Q27 lattice model,
     * which includes 27 discrete velocity directions in 3D space. It contains:
     * - Velocity components (cx, cy, cz) for each direction
     * - Weight coefficients for each direction
     * - Methods for moment calculation and population reconstruction
     * - Equilibrium distribution functions
     **/
    class D3Q27 : private velocitySet
    {
    public:
        using vs = constants::D3Q27;

        /**
         * @brief Default constructor (consteval)
         **/
        __device__ __host__ [[nodiscard]] inline consteval D3Q27(){};

        /**
         * @brief Get number of discrete velocity directions
         * @return 27 (number of directions in D3Q27 lattice)
         **/
        __device__ __host__ [[nodiscard]] static inline consteval label_t Q() noexcept
        {
            return vs::Q();
        }

        /**
         * @brief Get number of velocity components on a lattice face
         * @return 9 (number of directions crossing each face in D3Q27)
         **/
        __device__ __host__ [[nodiscard]] static inline consteval label_t QF() noexcept
        {
            return vs::QF();
        }

        /**
         * @brief Get weight for stationary component (q=0)
         **/
        template <typename T>
        __device__ __host__ [[nodiscard]] static inline consteval T w_0() noexcept
        {
            return static_cast<T>(static_cast<double>(8) / static_cast<double>(27));
        }

        /**
         * @brief Get weight for orthogonal directions (q=1-6)
         **/
        template <typename T>
        __device__ __host__ [[nodiscard]] static inline consteval T w_1() noexcept
        {
            return static_cast<T>(static_cast<double>(2) / static_cast<double>(27));
        }

        /**
         * @brief Get weight for diagonal directions (q=7-18)
         **/
        template <typename T>
        __device__ __host__ [[nodiscard]] static inline consteval T w_2() noexcept
        {
            return static_cast<T>(static_cast<double>(1) / static_cast<double>(54));
        }

        /**
         * @brief Get weight for diagonal directions (q=19-26)
         **/
        template <typename T>
        __device__ __host__ [[nodiscard]] static inline consteval T w_3() noexcept
        {
            return static_cast<T>(static_cast<double>(1) / static_cast<double>(216));
        }

        /**
         * @brief Get all weights for computation
         * @return Thread array of 27 weights in D3Q27 order
         **/
        template <typename T>
        __device__ __host__ [[nodiscard]] static inline consteval const thread::array<T, vs::Q()> w_q() noexcept
        {
            return {
                w_0<T>(),
                w_1<T>(), w_1<T>(), w_1<T>(), w_1<T>(), w_1<T>(), w_1<T>(),
                w_2<T>(), w_2<T>(), w_2<T>(), w_2<T>(), w_2<T>(), w_2<T>(), w_2<T>(), w_2<T>(), w_2<T>(), w_2<T>(), w_2<T>(), w_2<T>(),
                w_3<T>(), w_3<T>(), w_3<T>(), w_3<T>(), w_3<T>(), w_3<T>(), w_3<T>(), w_3<T>()};
        }

        /**
         * @brief Get weight for specific direction
         * @tparam q_ Direction index (0-26)
         * @param[in] q Direction index as compile-time constant
         * @return Weight for specified direction
         **/
        template <typename T, const label_t q_>
        __device__ __host__ [[nodiscard]] static inline consteval T w_q(const q_i<q_> q) noexcept
        {
            // Check that we are accessing a valid member
            static_assert(q() < vs::Q(), "Invalid velocity set index in member function w(q)");

            // Return the component
            return w_q<T>()[q];
        }

        /**
         * @brief Get x-components for all directions
         * @return Thread array of 27 x-velocity components
         **/
        template <typename T>
        __device__ __host__ [[nodiscard]] static inline consteval const thread::array<T, vs::Q()> cx() noexcept
        {
            return {static_cast<T>(0), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(-1), static_cast<T>(1)};
        }

        /**
         * @brief Get x-component for specific direction
         * @tparam q_ Direction index (0-26)
         * @param[in] q Direction index as compile-time constant
         * @return x-component for specified direction
         **/
        template <typename T, const label_t q_>
        __device__ __host__ [[nodiscard]] static inline consteval T cx(const q_i<q_> q) noexcept
        {
            // Check that we are accessing a valid member
            static_assert(q() < vs::Q(), "Invalid velocity set index in member function cx(q)");

            // Return the component
            return cx<T>()[q];
        }

        /**
         * @brief Get y-components for all directions
         * @return Thread array of 27 y-velocity components
         **/
        template <typename T>
        __device__ __host__ [[nodiscard]] static inline consteval const thread::array<T, vs::Q()> cy() noexcept
        {
            return {static_cast<T>(0), static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(-1), static_cast<T>(1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(-1), static_cast<T>(1), static_cast<T>(1), static_cast<T>(-1)};
        }

        /**
         * @brief Get y-component for specific direction
         * @tparam q_ Direction index (0-26)
         * @param[in] q Direction index as compile-time constant
         * @return y-component for specified direction
         **/
        template <typename T, const label_t q_>
        __device__ __host__ [[nodiscard]] static inline consteval T cy(const q_i<q_> q) noexcept
        {
            // Check that we are accessing a valid member
            static_assert(q() < vs::Q(), "Invalid velocity set index in member function cy(q)");

            // Return the component
            return cy<T>()[q];
        }

        /**
         * @brief Get z-components for all directions
         * @return Thread array of 27 z-velocity components
         **/
        template <typename T>
        __device__ __host__ [[nodiscard]] static inline consteval const thread::array<T, vs::Q()> cz() noexcept
        {
            return {static_cast<T>(0), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(-1), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(1), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(-1), static_cast<T>(1), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(1), static_cast<T>(-1)};
        }

        /**
         * @brief Get z-component for specific direction
         * @tparam q_ Direction index (0-26)
         * @param[in] q Direction index as compile-time constant
         * @return z-component for specified direction
         **/
        template <typename T, const label_t q_>
        __device__ __host__ [[nodiscard]] static inline consteval T cz(const q_i<q_> q) noexcept
        {
            // Check that we are accessing a valid member
            static_assert(q() < vs::Q(), "Invalid velocity set index in member function cz(q)");

            // Return the component
            return cz<T>()[q];
        }

        /**
         * @brief Get alpha-components for all directions
         * @return Thread array of 19 alpha-velocity components
         **/
        template <typename T, const axis::direction alpha>
        __device__ __host__ [[nodiscard]] static inline consteval const thread::array<T, vs::Q()> c() noexcept
        {
            assertions::validate_direction<alpha, axis::CAN_BE_NULL>();

            if constexpr (alpha == axis::NO_DIRECTION)
            {
                thread::array<T, vs::Q()> result;
                for (std::size_t i = 0; i < vs::Q(); i++)
                {
                    result[i] = 1;
                }
                return result;
            }
            if constexpr (alpha == axis::X)
            {
                return cx<T>();
            }
            if constexpr (alpha == axis::Y)
            {
                return cy<T>();
            }
            if constexpr (alpha == axis::Z)
            {
                return cz<T>();
            }
        }

        /**
         * @brief Reconstruct population distribution from moments (in-place)
         * @param[out] pop Population array to be filled
         * @param[in] moments Moment array (10 components)
         **/
        __device__ static inline void reconstruct(thread::array<scalar_t, vs::Q()> &pop, const thread::array<scalar_t, NUMBER_MOMENTS()> &moments) noexcept
        {
            const scalar_t pics2 = static_cast<scalar_t>(1.0) - cs2<scalar_t>() * (moments[m_i<4>()] + moments[m_i<7>()] + moments[m_i<9>()]);

            const scalar_t rhow_0 = moments[m_i<0>()] * w_0<scalar_t>();
            const scalar_t rhow_1 = moments[m_i<0>()] * w_1<scalar_t>();
            const scalar_t rhow_2 = moments[m_i<0>()] * w_2<scalar_t>();
            const scalar_t rhow_3 = moments[m_i<0>()] * w_3<scalar_t>();

            pop[q_i<0>()] = rhow_0 * pics2;

            pop[q_i<1>()] = rhow_1 * (pics2 + moments[m_i<1>()] + moments[m_i<4>()]);
            pop[q_i<2>()] = rhow_1 * (pics2 - moments[m_i<1>()] + moments[m_i<4>()]);
            pop[q_i<3>()] = rhow_1 * (pics2 + moments[m_i<2>()] + moments[m_i<7>()]);
            pop[q_i<4>()] = rhow_1 * (pics2 - moments[m_i<2>()] + moments[m_i<7>()]);
            pop[q_i<5>()] = rhow_1 * (pics2 + moments[m_i<3>()] + moments[m_i<9>()]);
            pop[q_i<6>()] = rhow_1 * (pics2 - moments[m_i<3>()] + moments[m_i<9>()]);

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

            pop[q_i<19>()] = rhow_3 * (pics2 + moments[m_i<1>()] + moments[m_i<2>()] + moments[m_i<3>()] + moments[m_i<4>()] + moments[m_i<7>()] + moments[m_i<9>()] + (moments[m_i<5>()] + moments[m_i<6>()] + moments[m_i<8>()]));
            pop[q_i<20>()] = rhow_3 * (pics2 - moments[m_i<1>()] - moments[m_i<2>()] - moments[m_i<3>()] + moments[m_i<4>()] + moments[m_i<7>()] + moments[m_i<9>()] + (moments[m_i<5>()] + moments[m_i<6>()] + moments[m_i<8>()]));
            pop[q_i<21>()] = rhow_3 * (pics2 + moments[m_i<1>()] + moments[m_i<2>()] - moments[m_i<3>()] + moments[m_i<4>()] + moments[m_i<7>()] + moments[m_i<9>()] + (moments[m_i<5>()] - moments[m_i<6>()] - moments[m_i<8>()]));
            pop[q_i<22>()] = rhow_3 * (pics2 - moments[m_i<1>()] - moments[m_i<2>()] + moments[m_i<3>()] + moments[m_i<4>()] + moments[m_i<7>()] + moments[m_i<9>()] + (moments[m_i<5>()] - moments[m_i<6>()] - moments[m_i<8>()]));
            pop[q_i<23>()] = rhow_3 * (pics2 + moments[m_i<1>()] - moments[m_i<2>()] + moments[m_i<3>()] + moments[m_i<4>()] + moments[m_i<7>()] + moments[m_i<9>()] - (moments[m_i<5>()] - moments[m_i<6>()] + moments[m_i<8>()]));
            pop[q_i<24>()] = rhow_3 * (pics2 - moments[m_i<1>()] + moments[m_i<2>()] - moments[m_i<3>()] + moments[m_i<4>()] + moments[m_i<7>()] + moments[m_i<9>()] - (moments[m_i<5>()] - moments[m_i<6>()] + moments[m_i<8>()]));
            pop[q_i<25>()] = rhow_3 * (pics2 - moments[m_i<1>()] + moments[m_i<2>()] + moments[m_i<3>()] + moments[m_i<4>()] + moments[m_i<7>()] + moments[m_i<9>()] - (moments[m_i<5>()] + moments[m_i<6>()] - moments[m_i<8>()]));
            pop[q_i<26>()] = rhow_3 * (pics2 + moments[m_i<1>()] - moments[m_i<2>()] - moments[m_i<3>()] + moments[m_i<4>()] + moments[m_i<7>()] + moments[m_i<9>()] - (moments[m_i<5>()] + moments[m_i<6>()] - moments[m_i<8>()]));
        }

        /**
         * @brief Reconstruct population distribution from moments (return)
         * @param[in] moments Moment array (10 components)
         * @return Population array with 27 components
         **/
        __device__ __host__ [[nodiscard]] static inline thread::array<scalar_t, vs::Q()> reconstruct(const thread::array<scalar_t, NUMBER_MOMENTS()> &moments) noexcept
        {
            const scalar_t pics2 = static_cast<scalar_t>(1.0) - cs2<scalar_t>() * (moments[m_i<4>()] + moments[m_i<7>()] + moments[m_i<9>()]);

            const scalar_t rhow_0 = moments[m_i<0>()] * w_0<scalar_t>();
            const scalar_t rhow_1 = moments[m_i<0>()] * w_1<scalar_t>();
            const scalar_t rhow_2 = moments[m_i<0>()] * w_2<scalar_t>();
            const scalar_t rhow_3 = moments[m_i<0>()] * w_3<scalar_t>();

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
                rhow_2 * (pics2 - moments[m_i<2>()] + moments[m_i<3>()] + moments[m_i<7>()] + moments[m_i<9>()] - moments[m_i<8>()]),
                rhow_3 * (pics2 + moments[m_i<1>()] + moments[m_i<2>()] + moments[m_i<3>()] + moments[m_i<4>()] + moments[m_i<7>()] + moments[m_i<9>()] + (moments[m_i<5>()] + moments[m_i<6>()] + moments[m_i<8>()])),
                rhow_3 * (pics2 - moments[m_i<1>()] - moments[m_i<2>()] - moments[m_i<3>()] + moments[m_i<4>()] + moments[m_i<7>()] + moments[m_i<9>()] + (moments[m_i<5>()] + moments[m_i<6>()] + moments[m_i<8>()])),
                rhow_3 * (pics2 + moments[m_i<1>()] + moments[m_i<2>()] - moments[m_i<3>()] + moments[m_i<4>()] + moments[m_i<7>()] + moments[m_i<9>()] + (moments[m_i<5>()] - moments[m_i<6>()] - moments[m_i<8>()])),
                rhow_3 * (pics2 - moments[m_i<1>()] - moments[m_i<2>()] + moments[m_i<3>()] + moments[m_i<4>()] + moments[m_i<7>()] + moments[m_i<9>()] + (moments[m_i<5>()] - moments[m_i<6>()] - moments[m_i<8>()])),
                rhow_3 * (pics2 + moments[m_i<1>()] - moments[m_i<2>()] + moments[m_i<3>()] + moments[m_i<4>()] + moments[m_i<7>()] + moments[m_i<9>()] - (moments[m_i<5>()] - moments[m_i<6>()] + moments[m_i<8>()])),
                rhow_3 * (pics2 - moments[m_i<1>()] + moments[m_i<2>()] - moments[m_i<3>()] + moments[m_i<4>()] + moments[m_i<7>()] + moments[m_i<9>()] - (moments[m_i<5>()] - moments[m_i<6>()] + moments[m_i<8>()])),
                rhow_3 * (pics2 - moments[m_i<1>()] + moments[m_i<2>()] + moments[m_i<3>()] + moments[m_i<4>()] + moments[m_i<7>()] + moments[m_i<9>()] - (moments[m_i<5>()] + moments[m_i<6>()] - moments[m_i<8>()])),
                rhow_3 * (pics2 + moments[m_i<1>()] - moments[m_i<2>()] - moments[m_i<3>()] + moments[m_i<4>()] + moments[m_i<7>()] + moments[m_i<9>()] - (moments[m_i<5>()] + moments[m_i<6>()] - moments[m_i<8>()]))};
        }

        /**
         * @brief Print velocity set information to terminal
         **/
        __host__ static void print() noexcept
        {
            std::cout << "D3Q27 {w, cx, cy, cz}:" << std::endl;
            std::cout << "{" << std::endl;
            printAll();
            std::cout << "};" << std::endl;
            std::cout << std::endl;
        }

    private:
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
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
Authors: Nathan Duggins, Breno Gemelgo (Geoenergia Lab, UDESC)

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
    Definition of the D3Q7 velocity set

Namespace
    LBM

SourceFiles
    D3Q7.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_D3Q7_CUH
#define __MBLBM_D3Q7_CUH

#include "velocitySet.cuh"

namespace LBM
{
    namespace constants
    {
        struct D3Q7
        {
            /**
             * @brief Get number of discrete velocity directions
             * @return 7 (number of directions in D3Q19 lattice)
             **/
            __device__ __host__ [[nodiscard]] static inline consteval label_t Q() noexcept
            {
                return 7;
            }

            /**
             * @brief Get number of velocity components on a lattice face
             * @return 1 (number of directions crossing each face in D3Q19)
             **/
            __device__ __host__ [[nodiscard]] static inline consteval label_t QF() noexcept
            {
                return 1;
            }
        };
    }

    /**
     * @class D3Q7
     * @brief Implements the D3Q7 velocity set for 3D Lattice Boltzmann simulations
     * @extends velocitySet
     *
     * This class provides the specific implementation for the D3Q7 lattice model,
     * which includes 7 discrete velocity directions in 3D space. It contains:
     * - Velocity components (cx, cy, cz) for each direction
     * - Weight coefficients for each direction
     **/
    class D3Q7 : private velocitySet
    {
    public:
        using vs = constants::D3Q7;

        /**
         * @brief Default constructor (consteval)
         **/
        __device__ __host__ [[nodiscard]] inline consteval D3Q7(){};

        /**
         * @brief Multiphase trait
         **/
        __device__ __host__ [[nodiscard]] static inline consteval bool isPhaseField() noexcept
        {
            return true;
        }

        /**
         * @brief Get number of discrete velocity directions
         * @return 7 (number of directions in D3Q7 lattice)
         **/
        __device__ __host__ [[nodiscard]] static inline consteval label_t Q() noexcept
        {
            return vs::Q();
        }

        /**
         * @brief Get number of velocity components on a lattice face
         * @return 1 (number of directions crossing each face in D3Q7)
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
            return static_cast<T>(static_cast<double>(1) / static_cast<double>(4));
        }

        /**
         * @brief Get weight for orthogonal directions (q=1-6)
         **/
        template <typename T>
        __device__ __host__ [[nodiscard]] static inline consteval T w_1() noexcept
        {
            return static_cast<T>(static_cast<double>(1) / static_cast<double>(8));
        }

        /**
         * @brief Get all weights for device computation
         * @return Thread array of 7 weights in D3Q7 order
         **/
        template <typename T>
        __device__ __host__ [[nodiscard]] static inline consteval const thread::array<T, 7> w_q() noexcept
        {
            // Return the component
            return {w_0<T>(), w_1<T>(), w_1<T>(), w_1<T>(), w_1<T>(), w_1<T>(), w_1<T>()};
        }

        /**
         * @brief Get weight for specific direction
         * @tparam q_ Direction index (0-6)
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
         * @brief Get x-components for all directions (device version)
         * @return Thread array of 7 x-velocity components
         **/
        template <typename T>
        __device__ __host__ [[nodiscard]] static inline consteval const thread::array<T, vs::Q()> cx() noexcept
        {
            // Return the component
            return {static_cast<T>(0), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0)};
        }

        /**
         * @brief Get x-component for specific direction
         * @tparam q_ Direction index (0-6)
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
         * @brief Get y-components for all directions (device version)
         * @return Thread array of 7 y-velocity components
         **/
        template <typename T>
        __device__ __host__ [[nodiscard]] static inline consteval const thread::array<T, vs::Q()> cy() noexcept
        {
            // Return the component
            return {static_cast<T>(0), static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(0), static_cast<T>(0)};
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
            static_assert(q() < vs::Q(), "Invalid velocity set index in member function cy(q)");

            // Return the component
            return cy<T>()[q];
        }

        /**
         * @brief Get z-components for all directions (device version)
         * @return Thread array of 7 z-velocity components
         **/
        template <typename T>
        __device__ __host__ [[nodiscard]] static inline consteval const thread::array<T, vs::Q()> cz() noexcept
        {
            // Return the component
            return {static_cast<T>(0), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(-1)};
        }

        /**
         * @brief Get z-component for specific direction
         * @tparam q_ Direction index (0-6)
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
        template <typename T, const axis::type alpha>
        __device__ __host__ [[nodiscard]] static inline consteval const thread::array<T, vs::Q()> c() noexcept
        {
            assertions::axis::validate<alpha, axis::CAN_BE_NULL>();

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
         * @param[in] moments Moment array (11 components)
         **/
        __device__ static inline void reconstruct(thread::array<scalar_t, vs::Q()> &pop, const thread::array<scalar_t, NUMBER_MOMENTS<true>()> &moments) noexcept
        {
            const scalar_t phiw_0 = moments[m_i<10>()] * w_0<scalar_t>();
            const scalar_t phiw_1 = moments[m_i<10>()] * w_1<scalar_t>();

            pop[q_i<0>()] = phiw_0;
            pop[q_i<1>()] = phiw_1 * (static_cast<scalar_t>(1) + velocitySet::unscale_i<scalar_t>() * moments[m_i<1>()]);
            pop[q_i<2>()] = phiw_1 * (static_cast<scalar_t>(1) - velocitySet::unscale_i<scalar_t>() * moments[m_i<1>()]);
            pop[q_i<3>()] = phiw_1 * (static_cast<scalar_t>(1) + velocitySet::unscale_i<scalar_t>() * moments[m_i<2>()]);
            pop[q_i<4>()] = phiw_1 * (static_cast<scalar_t>(1) - velocitySet::unscale_i<scalar_t>() * moments[m_i<2>()]);
            pop[q_i<5>()] = phiw_1 * (static_cast<scalar_t>(1) + velocitySet::unscale_i<scalar_t>() * moments[m_i<3>()]);
            pop[q_i<6>()] = phiw_1 * (static_cast<scalar_t>(1) - velocitySet::unscale_i<scalar_t>() * moments[m_i<3>()]);
        }

        /**
         * @brief Reconstruct population distribution from moments (return)
         * @param[in] moments Moment array (11 components)
         * @return Population array with 7 components
         **/
        __device__ __host__ [[nodiscard]] static inline thread::array<scalar_t, vs::Q()> reconstruct(const thread::array<scalar_t, NUMBER_MOMENTS<true>()> &moments) noexcept
        {
            thread::array<scalar_t, 7> pop;

            const scalar_t phiw_0 = moments[m_i<10>()] * w_0<scalar_t>();
            const scalar_t phiw_1 = moments[m_i<10>()] * w_1<scalar_t>();

            pop[q_i<0>()] = phiw_0;
            pop[q_i<1>()] = phiw_1 * (scalar_t(1) + velocitySet::unscale_i<scalar_t>() * moments[m_i<1>()]);
            pop[q_i<2>()] = phiw_1 * (scalar_t(1) - velocitySet::unscale_i<scalar_t>() * moments[m_i<1>()]);
            pop[q_i<3>()] = phiw_1 * (scalar_t(1) + velocitySet::unscale_i<scalar_t>() * moments[m_i<2>()]);
            pop[q_i<4>()] = phiw_1 * (scalar_t(1) - velocitySet::unscale_i<scalar_t>() * moments[m_i<2>()]);
            pop[q_i<5>()] = phiw_1 * (scalar_t(1) + velocitySet::unscale_i<scalar_t>() * moments[m_i<3>()]);
            pop[q_i<6>()] = phiw_1 * (scalar_t(1) - velocitySet::unscale_i<scalar_t>() * moments[m_i<3>()]);

            return pop;
        }

        /**
         * @brief Sharpen interface with a compressive term (in-place)
         * @param[out] pop Population array to be sharpened
         * @param[in] phi Phase field
         * @param[in] normx X-component of the unit interface normal
         * @param[in] normy Y-component of the unit interface normal
         * @param[in] normz Z-component of the unit interface normal
         **/
        __device__ static inline void sharpen(thread::array<scalar_t, 7> &pop, const scalar_t phi, const scalar_t normx, const scalar_t normy, const scalar_t normz) noexcept
        {
            const scalar_t sharp = w_1<scalar_t>() * device::gamma * phi * (scalar_t(1) - phi);

            pop[q_i<1>()] += sharp * normx;
            pop[q_i<2>()] -= sharp * normx;
            pop[q_i<3>()] += sharp * normy;
            pop[q_i<4>()] -= sharp * normy;
            pop[q_i<5>()] += sharp * normz;
            pop[q_i<6>()] -= sharp * normz;
        }

        /**
         * @brief Calculate phi from population distribution
         * @param[in] pop Population array (7 components)
         * @param[out] moments Moment array to be filled (11 components)
         **/
        __device__ inline static void calculate_phi(const thread::array<scalar_t, vs::Q()> &pop, thread::array<scalar_t, NUMBER_MOMENTS<true>()> &moments) noexcept
        {
            moments[m_i<10>()] = pop[q_i<0>()] + pop[q_i<1>()] + pop[q_i<2>()] + pop[q_i<3>()] + pop[q_i<4>()] + pop[q_i<5>()] + pop[q_i<6>()];
        }

        /**
         * @brief Print velocity set information to terminal
         **/
        __host__ static void print() noexcept
        {
            std::cout << "D3Q7 {w, cx, cy, cz}:" << std::endl;
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
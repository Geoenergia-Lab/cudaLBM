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
#include "../typedefs/typedefs.cuh"
#include "../globalFunctions.cuh"
#include "../array/threadArray.cuh"

namespace LBM
{
    /**
     * @brief Enumerated type for indexing pointers to halos
     **/
    typedef enum thermalModelEnum : bool
    {
        Thermal = 0,
        Isothermal = 1
    } thermalModel_t;

    template <const thermalModel_t IsothermalModel>
    class D3Q19;

    template <const thermalModel_t IsothermalModel>
    class D3Q27;

    namespace assertions
    {
        namespace velocitySet
        {
            /**
             * @brief Asserts that VelocitySet is a valid velocity set (D3Q19 or D3Q27)
             * @tparam VelocitySet The velocity set (D3Q19 or D3Q27)
             **/
            template <class VelocitySet>
            __device__ __host__ inline consteval void validate() noexcept
            {
                static_assert(((std::is_same<VelocitySet, D3Q19<Thermal>>::value) || (std::is_same<VelocitySet, D3Q27<Thermal>>::value) || (std::is_same<VelocitySet, D3Q19<Isothermal>>::value) || (std::is_same<VelocitySet, D3Q27<Isothermal>>::value)), "VelocitySet must be D3Q19 or D3Q27.");
            }
        }
    }

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
        __device__ __host__ [[nodiscard]] inline consteval velocitySet() noexcept {}

        /**
         * @brief Get the a^2 constant (3.0)
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

        __device__ __host__ [[nodiscard]] static inline constexpr const thread::array<scalar_t, 3> diagonal_term(
            const thread::array<scalar_t, NUMBER_MOMENTS()> &moments) noexcept
        {
            const scalar_t Delta_m = (moments[q_i<1>()] * moments[q_i<1>()] + moments[q_i<2>()] * moments[q_i<2>()] + moments[q_i<3>()] * moments[q_i<3>()] - moments[q_i<4>()] - moments[q_i<7>()] - moments[q_i<9>()]) / static_cast<scalar_t>(3);

            return {moments[q_i<4>()] + Delta_m, moments[q_i<7>()] + Delta_m, moments[q_i<9>()] + Delta_m};
        }

        /**
         * @brief Calculate a specific moment of the distribution function
         * @tparam VelocitySet The velocity set (D3Q19 or D3Q27)
         * @tparam alpha The first axis direction (X, Y, or Z)
         * @tparam beta The second axis direction (X, Y, or Z)
         * @param[in] pop The distribution function array
         * @return The calculated moment value
         **/
        template <class VelocitySet, const axis::type alpha, const axis::type beta>
        __device__ __host__ [[nodiscard]] static inline constexpr scalar_t calculate_moment(const thread::array<scalar_t, VelocitySet::Q()> &pop) noexcept
        {
            constexpr const thread::array<int, VelocitySet::Q()> c_AB = c_AlphaBeta<VelocitySet, alpha, beta>();
            constexpr const host::label_t N = number_non_zero(c_AB);
            constexpr const thread::array<int, N> C = non_zero_values<N>(c_AB);
            constexpr const thread::array<host::label_t, N> indices = non_zero_indices<N>(c_AB);

            return [&]<const host::label_t... Is>(std::index_sequence<Is...>)
            {
                return (process_momentum_element<C[Is]>(pop[indices[Is]]) + ...);
            }(std::make_index_sequence<N>{});
        }

        /**
         * @brief Calculate a specific moment of the distribution function
         * @tparam VelocitySet The velocity set (D3Q19 or D3Q27)
         * @tparam alpha The first axis direction (X, Y, or Z)
         * @tparam beta The second axis direction (X, Y, or Z)
         * @tparam BoundaryNormal The boundary normal vector type
         * @param[in] pop The distribution function array
         * @param[in] boundaryNormal Normal vector information at boundary node
         * @return The calculated moment value
         **/
        template <class VelocitySet, const axis::type alpha, const axis::type beta, class BoundaryNormal>
        __device__ __host__ [[nodiscard]] static inline constexpr scalar_t calculate_moment(const thread::array<scalar_t, VelocitySet::Q()> &pop, const BoundaryNormal &boundaryNormal) noexcept
        {
            constexpr const thread::array<int, VelocitySet::Q()> c_AB = c_AlphaBeta<VelocitySet, alpha, beta>();
            constexpr const host::label_t N = number_non_zero(c_AB);
            constexpr const thread::array<int, N> C = non_zero_values<N>(c_AB);
            constexpr const thread::array<host::label_t, N> indices = non_zero_indices<N>(c_AB);

            return [&]<const host::label_t... Is>(std::index_sequence<Is...>)
            {
                return (process_momentum_element<C[Is], VelocitySet, indices[Is]>(pop[indices[Is]], boundaryNormal) + ...);
            }(std::make_index_sequence<N>{});
        }

        /**
         * @brief Calculate all moments of the distribution function
         * @tparam VelocitySet The velocity set (D3Q19 or D3Q27)
         * @param[in] pop The distribution function array
         * @param[out] mom The calculated moments array
         **/
        template <class VelocitySet>
        __device__ __host__ static inline void calculate_moments(const thread::array<scalar_t, VelocitySet::Q()> &pop, thread::array<scalar_t, NUMBER_MOMENTS()> &mom) noexcept
        {
            // Density
            mom[m_i<0>()] = calculate_moment<VelocitySet, axis::NO_DIRECTION, axis::NO_DIRECTION>(pop);
            const scalar_t inv_rho = static_cast<scalar_t>(1) / mom[m_i<0>()];

            // Velocity
            mom[m_i<1>()] = calculate_moment<VelocitySet, axis::X, axis::NO_DIRECTION>(pop) * inv_rho;
            mom[m_i<2>()] = calculate_moment<VelocitySet, axis::Y, axis::NO_DIRECTION>(pop) * inv_rho;
            mom[m_i<3>()] = calculate_moment<VelocitySet, axis::Z, axis::NO_DIRECTION>(pop) * inv_rho;

            // Second order moments
            mom[m_i<4>()] = (calculate_moment<VelocitySet, axis::X, axis::X>(pop) * inv_rho) - cs2<scalar_t>();
            mom[m_i<5>()] = calculate_moment<VelocitySet, axis::X, axis::Y>(pop) * inv_rho;
            mom[m_i<6>()] = calculate_moment<VelocitySet, axis::X, axis::Z>(pop) * inv_rho;
            mom[m_i<7>()] = (calculate_moment<VelocitySet, axis::Y, axis::Y>(pop) * inv_rho) - cs2<scalar_t>();
            mom[m_i<8>()] = calculate_moment<VelocitySet, axis::Y, axis::Z>(pop) * inv_rho;
            mom[m_i<9>()] = (calculate_moment<VelocitySet, axis::Z, axis::Z>(pop) * inv_rho) - cs2<scalar_t>();
        }

        /**
         * @brief Calculate all moments of the distribution function
         * @tparam VelocitySet The velocity set (D3Q19 or D3Q27)
         * @tparam BoundaryNormal The boundary normal vector type
         * @param[in] pop The distribution function array
         * @param[out] mom The calculated moments array
         * @param[in] boundaryNormal Normal vector information at boundary node
         **/
        template <class VelocitySet, class BoundaryNormal>
        __device__ __host__ static inline void calculate_moments(const thread::array<scalar_t, VelocitySet::Q()> &pop, thread::array<scalar_t, NUMBER_MOMENTS()> &mom, const BoundaryNormal &boundaryNormal) noexcept
        {
            // Density
            mom[m_i<0>()] = calculate_moment<VelocitySet, axis::NO_DIRECTION, axis::NO_DIRECTION>(pop, boundaryNormal);
            const scalar_t inv_rho = static_cast<scalar_t>(1) / mom[m_i<0>()];

            // Velocity
            mom[m_i<1>()] = calculate_moment<VelocitySet, axis::X, axis::NO_DIRECTION>(pop, boundaryNormal) * inv_rho;
            mom[m_i<2>()] = calculate_moment<VelocitySet, axis::Y, axis::NO_DIRECTION>(pop, boundaryNormal) * inv_rho;
            mom[m_i<3>()] = calculate_moment<VelocitySet, axis::Z, axis::NO_DIRECTION>(pop, boundaryNormal) * inv_rho;

            // Second order moments
            mom[m_i<4>()] = (calculate_moment<VelocitySet, axis::X, axis::X>(pop, boundaryNormal) * inv_rho) - cs2<scalar_t>();
            mom[m_i<5>()] = calculate_moment<VelocitySet, axis::X, axis::Y>(pop, boundaryNormal) * inv_rho;
            mom[m_i<6>()] = calculate_moment<VelocitySet, axis::X, axis::Z>(pop, boundaryNormal) * inv_rho;
            mom[m_i<7>()] = (calculate_moment<VelocitySet, axis::Y, axis::Y>(pop, boundaryNormal) * inv_rho) - cs2<scalar_t>();
            mom[m_i<8>()] = calculate_moment<VelocitySet, axis::Y, axis::Z>(pop, boundaryNormal) * inv_rho;
            mom[m_i<9>()] = (calculate_moment<VelocitySet, axis::Z, axis::Z>(pop, boundaryNormal) * inv_rho) - cs2<scalar_t>();
        }

        /**
         * @brief Returns the indices of the distribution functions on a specific face
         * @tparam VelocitySet The velocity set (D3Q19 or D3Q27)
         * @tparam alpha The axis direction (X, Y, or Z)
         * @tparam coeff The value of the coordinate along the axis (-1 or 1)
         * @return Indices of the distribution on a specific face
         **/
        template <class VelocitySet, const axis::type alpha, const int coeff>
        __device__ __host__ [[nodiscard]] static inline consteval thread::array<host::label_t, VelocitySet::QF()> indices_on_face() noexcept
        {
            assertions::velocitySet::validate<VelocitySet>();
            axis::assertions::validate<alpha, axis::NOT_NULL>();

            velocityCoefficient::assertions::validate<coeff, velocityCoefficient::NOT_NULL>();

            constexpr const thread::array<int, VelocitySet::Q()> vals = VelocitySet::template c<int, alpha>();

            thread::array<host::label_t, VelocitySet::QF()> indices;

            host::label_t j = 0;

            for (host::label_t i = 0; i < VelocitySet::Q(); i++)
            {
                if (vals[i] == coeff)
                {
                    indices[j] = i;
                    j++;
                }
            }

            return indices;
        }

    private:
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
        template <typename T, class VelocitySet, class BoundaryNormal, const device::label_t q_>
        __device__ __host__ [[nodiscard]] static inline constexpr T is_incoming(const q_i<q_> q, const BoundaryNormal &boundaryNormal) noexcept
        {
            // boundaryNormal.x > 0  => EAST boundary
            // boundaryNormal.x < 0  => WEST boundary
            const bool cond_x = (boundaryNormal.isEast() & is_negative<VelocitySet, axis::X>(q)) | (boundaryNormal.isWest() & is_positive<VelocitySet, axis::X>(q));

            // boundaryNormal.y > 0  => NORTH boundary
            // boundaryNormal.y < 0  => SOUTH boundary
            const bool cond_y = (boundaryNormal.isNorth() & is_negative<VelocitySet, axis::Y>(q)) | (boundaryNormal.isSouth() & is_positive<VelocitySet, axis::Y>(q));

            // boundaryNormal.z > 0  => FRONT boundary
            // boundaryNormal.z < 0  => BACK boundary
            const bool cond_z = (boundaryNormal.isFront() & is_negative<VelocitySet, axis::Z>(q)) | (boundaryNormal.isBack() & is_positive<VelocitySet, axis::Z>(q));

            return static_cast<T>(!(cond_x | cond_y | cond_z));
        }

        /**
         * @brief Returns the product of the c values for two directions
         **/
        template <class VelocitySet, const axis::type alpha, const axis::type beta>
        __device__ __host__ [[nodiscard]] static inline consteval const thread::array<int, VelocitySet::Q()> c_AlphaBeta() noexcept
        {
            assertions::velocitySet::validate<VelocitySet>();
            axis::assertions::validate<alpha, axis::CAN_BE_NULL>();
            axis::assertions::validate<beta, axis::CAN_BE_NULL>();

            return VelocitySet::template c<int, alpha>() * VelocitySet::template c<int, beta>();
        }

        /**
         * @brief Adds or subtracts a particular population based on the sign of the coefficient
         * @tparam coeff The velocity set coefficient (-1 or 1)
         * @param[in] pop_value A particular population
         * @return Plus or minus pop_value depending on the value of coeff
         **/
        template <const int coeff>
        __device__ __host__ [[nodiscard]] static inline constexpr scalar_t process_momentum_element(
            const scalar_t pop_value) noexcept
        {
            velocityCoefficient::assertions::validate<coeff, velocityCoefficient::NOT_NULL>();

            if constexpr (coeff == 1)
            {
                return pop_value;
            }

            if constexpr (coeff == -1)
            {
                return -pop_value;
            }
        }

        /**
         * @brief Processes a momentum element for a specific coefficient
         **/
        /**
         * @brief Adds or subtracts a particular population based on the sign of the coefficient
         * @tparam coeff The velocity set coefficient (-1 or 1)
         * @tparam VelocitySet The velocity set (D3Q19 or D3Q27)
         * @tparam BoundaryNormal The boundary normal vector type
         * @param[in] pop_value A particular population
         * @param[in] boundaryNormal
         * @return Plus or minus pop_value depending on the value of coeff
         **/
        template <const int coeff, class VelocitySet, const device::label_t I, class BoundaryNormal>
        __device__ __host__ [[nodiscard]] static inline constexpr scalar_t process_momentum_element(
            const scalar_t pop_value,
            const BoundaryNormal &boundaryNormal) noexcept
        {
            velocityCoefficient::assertions::validate<coeff, velocityCoefficient::NOT_NULL>();

            if constexpr (coeff == 1)
            {
                return is_incoming<scalar_t, VelocitySet>(q_i<I>(), boundaryNormal) * pop_value;
            }

            if constexpr (coeff == -1)
            {
                return -is_incoming<scalar_t, VelocitySet>(q_i<I>(), boundaryNormal) * pop_value;
            }
        }

        /**
         * @brief Determines whether or not a particular lattice coefficient is negative
         * @tparam VelocitySet The velocity set (D3Q19 or D3Q27)
         * @tparam alpha The axis direction (X, Y, or Z)
         * @param[in] q The lattice index
         * @return True if the lattice coefficient is negative, false otherwise
         **/
        template <class VelocitySet, const axis::type alpha, const device::label_t q_>
        __device__ __host__ [[nodiscard]] static inline consteval bool is_negative(const q_i<q_> q) noexcept
        {
            assertions::velocitySet::validate<VelocitySet>();
            axis::assertions::validate<alpha, axis::NOT_NULL>();

            return (VelocitySet::template c<int, alpha>()[q] < 0);
        }

        /**
         * @brief Determines whether or not a particular lattice coefficient is positive
         * @tparam VelocitySet The velocity set (D3Q19 or D3Q27)
         * @tparam alpha The axis direction (X, Y, or Z)
         * @param[in] The lattice index
         * @return True if the lattice coefficient is positive, false otherwise
         **/
        template <class VelocitySet, const axis::type alpha, const device::label_t q_>
        __device__ __host__ [[nodiscard]] static inline consteval bool is_positive(const q_i<q_> q) noexcept
        {
            assertions::velocitySet::validate<VelocitySet>();
            axis::assertions::validate<alpha, axis::NOT_NULL>();

            return (VelocitySet::template c<int, alpha>()[q] > 0);
        }

    protected:
        /**
         * @brief Returns the string corresponding to a lattice velocity coefficient
         * @tparam coeff The velocity coefficient
         **/
        template <const int coeff>
        __host__ [[nodiscard]] static inline consteval const char *c()
        {
            if constexpr (coeff == 0)
            {
                return "0";
            }

            if constexpr (coeff == -1)
            {
                return "-1";
            }

            if constexpr (coeff == 1)
            {
                return "+1";
            }
        }
    };
}

#include "D3Q19.cuh"
#include "D3Q27.cuh"

#endif
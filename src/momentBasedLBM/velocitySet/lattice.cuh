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
    Definition of lattice speeds and weights for all 3D lattices

Namespace
    LBM

SourceFiles
    lattice.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_LATTICE_CUH
#define __MBLBM_LATTICE_CUH

namespace LBM
{
    template <const host::label_t Q_>
    class lattice
    {
        using This = lattice<Q_>;

    private:
        /**
         * @brief Valid lattice velocity set sizes (D3Q7, D3Q19, D3Q27)
         * @tparam T The underlying type of the array
         **/
        template <typename T>
        __device__ __host__ [[nodiscard]] static inline consteval const thread::array<T, 3> validQ() noexcept
        {
            return {static_cast<T>(7), static_cast<T>(19), static_cast<T>(27)};
        }

    public:
        /**
         * @brief Checks if the provided lattice velocity set size is valid (D3Q7, D3Q19, D3Q27)
         * @param[in] Q The lattice velocity set size to check
         * @return True if the provided lattice velocity set size is valid, false otherwise
         **/
        static_assert(validQ<host::label_t>().template contains<Q_>(), "Lattice velocity set must be D3Q7, D3Q19 or D3Q27.");

        /**
         * @brief Returns the number of unique weights in the velocity set
         * @return The number of unique weights in the velocity set
         **/
        __device__ __host__ static inline consteval host::label_t nPerm() noexcept
        {
            return static_cast<host::label_t>(1) + static_cast<host::label_t>(Q_ > 7) + static_cast<host::label_t>(Q_ > 19);
        }

        /**
         * @brief Calculates the weighted density for each population based on the total density
         * @param[in] rho The total density
         * @return An array of weighted densities for the populations
         **/
        __device__ __host__ static inline constexpr const thread::array<const scalar_t, This::nPerm()> rhow(const scalar_t rho) noexcept
        {
            if constexpr (Q_ == 7)
            {
                return {rho * This::template w_1<scalar_t>()};
            }
            if constexpr (Q_ == 19)
            {
                return {rho * This::template w_1<scalar_t>(), rho * This::template w_2<scalar_t>()};
            }
            if constexpr (Q_ == 27)
            {
                return {rho * This::template w_1<scalar_t>(), rho * This::template w_2<scalar_t>(), rho * This::template w_3<scalar_t>()};
            }
        }

        /**
         * @brief Calculates the weighted density for a specific population index
         * @tparam i The population index
         * @param[in] rho_w The array of weighted densities for the populations
         * @return The weighted density for the specified population index
         **/
        template <const host::label_t i>
        __device__ __host__ static inline constexpr scalar_t rhow(const thread::array<const scalar_t, This::nPerm()> &rho_w) noexcept
        {
            return rho_w[m_i<static_cast<host::label_t>(i >= 7) + static_cast<host::label_t>(i >= 19)>()];
        }

        /**
         * @brief Get weight for stationary component (q=0)
         * @tparam T The return type
         **/
        template <typename T>
        __device__ __host__ [[nodiscard]] static inline consteval T w_0() noexcept
        {
            if constexpr (Q_ == 27)
            {
                return static_cast<T>(static_cast<double>(8) / static_cast<double>(27));
            }

            if constexpr (Q_ == 19)
            {
                return static_cast<T>(static_cast<double>(1) / static_cast<double>(3));
            }
        }

        /**
         * @brief Get weight for orthogonal directions (q=1-6)
         * @tparam T The return type
         **/
        template <typename T>
        __device__ __host__ [[nodiscard]] static inline consteval T w_1() noexcept
        {
            if constexpr (Q_ == 27)
            {
                return static_cast<T>(static_cast<double>(2) / static_cast<double>(27));
            }

            if constexpr (Q_ == 19)
            {
                return static_cast<T>(static_cast<double>(1) / static_cast<double>(18));
            }
        }

        /**
         * @brief Get weight for diagonal directions (q=7-18)
         * @tparam T The return type
         **/
        template <typename T>
        __device__ __host__ [[nodiscard]] static inline consteval T w_2() noexcept
        {
            if constexpr (Q_ == 27)
            {
                return static_cast<T>(static_cast<double>(1) / static_cast<double>(54));
            }

            if constexpr (Q_ == 19)
            {
                return static_cast<T>(static_cast<double>(1) / static_cast<double>(36));
            }
        }

        /**
         * @brief Get weight for corner directions (q=19-26)
         * @tparam T The return type
         **/
        template <typename T>
        __device__ __host__ [[nodiscard]] static inline consteval T w_3() noexcept
        {
            if constexpr (Q_ == 27)
            {
                return static_cast<T>(static_cast<double>(1) / static_cast<double>(216));
            }

            if constexpr (Q_ == 19)
            {
                return static_cast<T>(0);
            }
        }

        /**
         * @brief Get all weights for device computation
         * @tparam T The underlying data type of the array
         * @return Thread array of 27 weights in D3Q27 order
         **/
        template <typename T>
        __device__ __host__ [[nodiscard]] static inline consteval const thread::array<T, Q_> w_q() noexcept
        {
            return make_first_Q<T>(w_impl<T>());
        }

        /**
         * @brief Get the lattice speeds as a thread::array
         * @tparam T The underlying data type of the array
         * @tparam alpha The axis (X, Y or Z)
         **/
        template <typename T, const axis::type alpha>
        __device__ __host__ [[nodiscard]] static inline consteval const thread::array<T, Q_> c() noexcept
        {
            return make_first_Q<T>(c_base_impl<T, alpha>());
        }

        /**
         * @brief Returns a component of the velocity set along an arbitrary axis
         * @tparam T The type of data to return
         * @tparam alpha The axis (X, Y or Z)
         * @tparam q_ Value of the lattice index
         * @param[in] q The lattice index
         **/
        template <typename T, const axis::type alpha, const device::label_t q_>
        __device__ __host__ [[nodiscard]] static inline consteval T c(const q_i<q_> q) noexcept
        {
            return c<T, alpha>()[q];
        }

        /**
         * @brief Returns the number of components of the velocity set
         * @tparam T The return type
         **/
        template <typename T = host::label_t>
        __device__ __host__ [[nodiscard]] static inline consteval T Q() noexcept
        {
            return Q_;
        }

        /**
         * @brief Returns the number of components of the velocity set facing any given cardinal direction
         * @tparam T The return type
         **/
        template <typename T>
        __device__ __host__ [[nodiscard]] static inline consteval T QF() noexcept
        {
            return make_first_Q<int>(cx_base<int>()).template count<1, true>();
        }

    private:
        /**
         * @brief Generic function to return the first N values of an array of arbitrary size
         * @tparam T The fundamental type of the underlying array
         * @tparam N The number of elements to return
         * @tparam M The size of the input array
         * @param[in] arr The input array of size M
         * @return A thread::array containing the first N elements of the input array
         **/
        template <typename T, const host::label_t N, const host::label_t M>
        __device__ __host__ [[nodiscard]] static inline consteval const thread::array<T, N> make_first_N(const thread::array<T, M> &arr) noexcept
        {
            return [&]<const host::label_t... Is>(const std::index_sequence<Is...>)
            {
                return thread::array<T, N>{arr[Is]...};
            }(std::make_index_sequence<N>{});
        }

        /**
         * @brief Returns the first Q_ elements of an arbitrary array
         * @tparam T The type of the underlying array
         * @param[in] arr The array of which to make the first Q_ elements
         * @return A thread::array containing the first Q_ elements of the input array
         **/
        template <typename T>
        __device__ __host__ [[nodiscard]] static inline consteval const thread::array<T, Q_> make_first_Q(const thread::array<T, 27> &arr) noexcept
        {
            return make_first_N<T, Q_>(arr);
        }

        /**
         * @brief Fundamental definition of the lattice speeds
         * @tparam T The type of the underlying array
         * @tparam alpha The axis direction (X, Y or Z)
         **/
        template <typename T, const axis::type alpha>
        __device__ __host__ [[nodiscard]] static inline consteval const thread::array<T, 27> c_base_impl() noexcept
        {
            axis::assertions::validate<alpha, axis::CAN_BE_NULL>();

            if constexpr (alpha == axis::NO_DIRECTION)
            {
                constexpr const std::integral_constant<T, static_cast<T>(1)> val;
                return thread::array<T, 27>(val);
            }
            if constexpr (alpha == axis::X)
            {
                return cx_base<T>();
            }
            if constexpr (alpha == axis::Y)
            {
                return cy_base<T>();
            }
            if constexpr (alpha == axis::Z)
            {
                return cz_base<T>();
            }
        }

        /**
         * @brief Get all weights for device computation
         * @tparam T The return type
         * @return Thread array of 27 weights in D3Q27 order
         **/
        template <typename T>
        __device__ __host__ [[nodiscard]] static inline consteval const thread::array<T, 27> w_impl() noexcept
        {
            return {w_0<T>(), w_1<T>(), w_1<T>(), w_1<T>(), w_1<T>(), w_1<T>(), w_1<T>(), w_2<T>(), w_2<T>(), w_2<T>(), w_2<T>(), w_2<T>(), w_2<T>(), w_2<T>(), w_2<T>(), w_2<T>(), w_2<T>(), w_2<T>(), w_2<T>(), w_3<T>(), w_3<T>(), w_3<T>(), w_3<T>(), w_3<T>(), w_3<T>(), w_3<T>(), w_3<T>()};
        }

        /**
         * @brief Get x-components for all directions
         * @tparam T The return type
         * @return Thread array of 27 x-velocity components
         **/
        template <typename T>
        __device__ __host__ [[nodiscard]] static inline consteval const thread::array<T, 27> cx_base() noexcept
        {
            return {static_cast<T>(0), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(-1), static_cast<T>(1)};
        }

        /**
         * @brief Get y-components for all directions
         * @tparam T The return type
         * @return Thread array of 27 y-velocity components
         **/
        template <typename T>
        __device__ __host__ [[nodiscard]] static inline consteval const thread::array<T, 27> cy_base() noexcept
        {
            return {static_cast<T>(0), static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(-1), static_cast<T>(1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(-1), static_cast<T>(1), static_cast<T>(1), static_cast<T>(-1)};
        }

        /**
         * @brief Get z-components for all directions
         * @tparam T The return type
         * @return Thread array of 27 z-velocity components
         **/
        template <typename T>
        __device__ __host__ [[nodiscard]] static inline consteval const thread::array<T, 27> cz_base() noexcept
        {
            return {static_cast<T>(0), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(-1), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(1), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(-1), static_cast<T>(1), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(1), static_cast<T>(-1)};
        }
    };
}

#endif
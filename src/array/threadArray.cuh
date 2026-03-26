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
    This file defines the thread array class, which is a fixed-size array
    container designed for use in single-threaded device code. The class
    provides compile-time bounds checking and supports various constructors for
    initializing the array with specific values or from global memory using a
    shared buffer cache. It also overloads basic arithmetic operators for
    element-wise operations and provides methods for accessing and modifying
    elements. The thread array is intended to be used within CUDA kernels where
    each thread manages its own small array of data, such as the distribution
    functions in a lattice Boltzmann simulation.

Namespace
    LBM

SourceFiles
    threadArray.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_THREADARRAY_CUH
#define __MBLBM_THREADARRAY_CUH

#include "../globalFunctions.cuh"

namespace LBM
{
    namespace thread
    {
        template <const host::label_t i, const host::label_t N>
        concept in_bounds = (i < N);

        /**
         * @brief Fixed-size array container for single-threaded device code
         * @tparam T Type of elements stored in the array
         * @tparam N Number of elements in the array (compile-time constant)
         **/
        template <typename T, const host::label_t N>
        class array
        {
        public:
            /**
             * @brief Constructs array with specified initial values
             * @tparam Args Variadic template parameter pack for initial values
             * @param[in] args Initial values for array elements
             * @pre Number of arguments must exactly match template parameter N
             * @note Compile-time enforced check ensures correct number of arguments
             **/
            template <typename... Args>
            __device__ __host__ [[nodiscard]] inline constexpr array(const Args... args) : data_{args...}
            {
                static_assert(sizeof...(Args) == N, "Incorrect number of arguments");
            }

            /**
             * @brief Fill constructor
             * @param[in] value Initial value for all array elements
             **/
            // template <std::enable_if_t<(N != 1), bool> = true>
            // __device__ __host__ [[nodiscard]] inline consteval array(const T value) noexcept
            // {
            //     device::constexpr_for<0, N>(
            //         [&](const auto i)
            //         {
            //             data_[q_i<i>()] = value;
            //         });
            // }

            /**
             * @brief Default constructor (value-initializes all elements)
             * @note Elements will be default-initialized or zero-initialized
             **/
            [[nodiscard]] inline consteval array() = default;

            /**
             * @brief Addition operator
             * @return The sum of two arrays of the same type and size
             **/
            __device__ __host__ [[nodiscard]] inline constexpr thread::array<T, N> operator+(const thread::array<T, N> &A) const __restrict__ noexcept
            {
                return [&]<const host::label_t... Is>(std::index_sequence<Is...>)
                {
                    return thread::array<T, N>{
                        (data_[size_constant<Is>{}] + A[size_constant<Is>{}])...};
                }(std::make_index_sequence<N>{});
            }

            /**
             * @brief Subtraction operator
             * @return The subtraction of two arrays of the same type and size
             **/
            __device__ __host__ [[nodiscard]] inline constexpr thread::array<T, N> operator-(const thread::array<T, N> &A) const __restrict__ noexcept
            {
                return [&]<const host::label_t... Is>(std::index_sequence<Is...>)
                {
                    return thread::array<T, N>{
                        (data_[size_constant<Is>{}] - A[size_constant<Is>{}])...};
                }(std::make_index_sequence<N>{});
            }

            /**
             * @brief Multiplication operator
             * @return The dot product of two arrays of the same type and size
             **/
            __device__ __host__ [[nodiscard]] inline constexpr thread::array<T, N> operator*(const thread::array<T, N> &A) const __restrict__ noexcept
            {
                return [&]<const host::label_t... Is>(std::index_sequence<Is...>)
                {
                    return thread::array<T, N>{
                        (data_[size_constant<Is>{}] * A[size_constant<Is>{}])...};
                }(std::make_index_sequence<N>{});
            }

            /**
             * @brief Division operator
             * @return The dot product of the first array and the inverse of the second, both of which are of the same type and size
             **/
            __device__ __host__ [[nodiscard]] inline constexpr thread::array<T, N> operator/(const thread::array<T, N> &A) const __restrict__ noexcept
            {
                return [&]<const host::label_t... Is>(std::index_sequence<Is...>)
                {
                    return thread::array<T, N>{
                        (data_[size_constant<Is>{}] / A[size_constant<Is>{}])...};
                }(std::make_index_sequence<N>{});
            }

            /**
             * @brief Compile-time mutable element access
             * @tparam index_ Compile-time index value
             * @param[in] index Index tag (label_constant wrapper)
             * @return Reference to element at specified index
             * @pre index_ must be in range [0, N-1]
             * @note No runtime bounds checking - compile-time safe
             **/
            template <const host::label_t index_>
            __device__ __host__ [[nodiscard]] inline constexpr T &operator[](const size_constant<index_> &index) __restrict__ noexcept
            {
                assert_legal_access<index_>();
                return data_[size_constant<index.value>()];
            }

            /**
             * @brief Compile-time read-only element access
             * @tparam index_ Compile-time index value
             * @param[in] index Index tag (label_constant wrapper)
             * @return Const reference to element at specified index
             * @pre index_ must be in range [0, N-1]
             * @note No runtime bounds checking - compile-time safe
             **/
            template <const host::label_t index_>
            __device__ __host__ [[nodiscard]] inline constexpr const T &operator[](const size_constant<index_> &index) __restrict__ const noexcept
            {
                assert_legal_access<index_>();
                return data_[size_constant<index.value>()];
            }

            /**
             * @brief Unified element access (compile-time or runtime)
             * @tparam Index Type of index (integral type or std::integral_constant)
             * @param[in] idx Index value or compile-time index tag
             * @return Reference to element at specified index
             * @pre Index must be in range [0, N-1]
             * @note Compile-time bounds checking for integral_constant types
             * @note Runtime access for integral types (no bounds checking)
             **/
            template <typename Idx>
            __device__ __host__ [[nodiscard]] inline constexpr T &operator[](const Idx idx) __restrict__ noexcept
            {
                // Runtime index
                return data_[idx];
            }

            /**
             * @brief Unified read-only element access (compile-time or runtime)
             * @tparam Index Type of index (integral type or std::integral_constant)
             * @param[in] idx Index value or compile-time index tag
             * @return Const reference to element at specified index
             * @pre Index must be in range [0, N-1]
             * @note Compile-time bounds checking for integral_constant types
             * @note Runtime access for integral types (no bounds checking)
             **/
            template <typename Idx>
            __device__ __host__ [[nodiscard]] inline constexpr const T &operator[](const Idx idx) __restrict__ const noexcept
            {
                return data_[idx];
            }

            /**
             * @brief Returns a pointer to the first element of the array
             * @return Pointer to data_[0]
             **/
            __device__ __host__ [[nodiscard]] inline constexpr const T *data() __restrict__ const noexcept
            {
                return &data_[0];
            }
            __device__ __host__ [[nodiscard]] inline constexpr T *data() __restrict__ noexcept
            {
                return &data_[0];
            }

            /**
             * @brief Returns the number of elements in the array
             * @return Compile-time constant number of elements (N)
             **/
            __device__ __host__ [[nodiscard]] static inline consteval host::label_t size() noexcept
            {
                return N;
            }

        private:
            /**
             * @brief The underlying data
             **/
            T ptrRestrict data_[N];

            /**
             * @brief Compile-time check that accesses are valid
             **/
            template <const host::label_t i>
            __device__ __host__ static inline consteval void assert_legal_access() noexcept
            {
                static_assert(in_bounds<i, N>, "index is out of range: Must be < N.");
            }
        };
    }

    /**
     * @brief Computes the number of non-zero elements of an array
     * @tparam T Type of elements in the array
     * @tparam N Size of the array
     * @param[in] arr The input array
     * @return Number of non-zero elements in the array
     **/
    template <typename T, const host::label_t N>
    __device__ __host__ [[nodiscard]] inline consteval host::label_t number_non_zero(const thread::array<T, N> &arr)
    {
        host::label_t n = 0;

        for (host::label_t i = 0; i < N; i++)
        {
            if (!(arr[i] == 0))
            {
                n++;
            }
        }

        return n;
    }

    /**
     * @brief Get the non-zero values in the array
     * @tparam ReturnSize Size of the returned array
     * @tparam T Type of elements in the array
     * @tparam N Size of the input array
     * @param[in] arr The input array
     * @return Array containing only non-zero values from the input array
     **/
    template <const host::label_t ReturnSize, typename T, const host::label_t N>
    __device__ __host__ [[nodiscard]] static inline constexpr thread::array<T, ReturnSize> non_zero_values(const thread::array<T, N> &arr) noexcept
    {
        thread::array<T, ReturnSize> coefficients{};

        host::label_t count = 0;

        for (host::label_t i = 0; i < N; i++)
        {
            if (arr[i] != 0)
            {
                coefficients[count] = arr[i];
                count++;
            }
        }

        return coefficients;
    }

    /**
     * @brief Get the non-zero indices in the array
     * @tparam ReturnSize Size of the returned array
     * @tparam T Type of elements in the array
     * @tparam N Size of the input array
     * @param[in] arr The input array
     * @return Array containing only non-zero indices from the input array
     **/
    template <const device::label_t ReturnSize, typename T, const host::label_t N>
    __device__ __host__ [[nodiscard]] static inline constexpr thread::array<host::label_t, ReturnSize> non_zero_indices(const thread::array<T, N> &arr) noexcept
    {
        thread::array<host::label_t, ReturnSize> indices{};

        host::label_t count = 0;

        for (host::label_t i = 0; i < N; i++)
        {
            if (arr[i] != 0)
            {
                indices[count] = i;
                count++;
            }
        }

        return indices;
    }
}

#endif

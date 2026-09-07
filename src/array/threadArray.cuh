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
             * @tparam v Value to fill the array
             * @param[in] value Initial value for all array elements
             **/
            template <const T v>
            __device__ __host__ [[nodiscard]] inline consteval array(const std::integral_constant<T, v> &value) noexcept
            {
                for (host::label_t i = 0; i < N; i++)
                {
                    data_[i] = value;
                }
            }

            /**
             * @brief Default constructor (value-initializes all elements)
             * @note Elements will be default-initialized or zero-initialized
             **/
            [[nodiscard]] inline consteval array() = default;
            __device__ __host__ [[nodiscard]] array(const array<T, N> &) = delete;
            __device__ __host__ [[nodiscard]] array &operator=(const array<T, N> &) = delete;

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
             * @brief Addition operator
             * @return The sum of the array and a constant of type T
             **/
            __device__ __host__ [[nodiscard]] inline constexpr thread::array<T, N> operator+(const T &A) const __restrict__ noexcept
            {
                return [&]<const host::label_t... Is>(std::index_sequence<Is...>)
                {
                    return thread::array<T, N>{
                        (data_[size_constant<Is>{}] + A)...};
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
             * @brief Subtraction operator
             * @return The subtraction of the array and a constant of type T
             **/
            __device__ __host__ [[nodiscard]] inline constexpr thread::array<T, N> operator-(const T &A) const __restrict__ noexcept
            {
                return [&]<const host::label_t... Is>(std::index_sequence<Is...>)
                {
                    return thread::array<T, N>{
                        (data_[size_constant<Is>{}] - A)...};
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
             * @brief Multiplication operator
             * @return The product of the array and a constant of type T
             **/
            __device__ __host__ [[nodiscard]] inline constexpr thread::array<T, N> operator*(const T &A) const __restrict__ noexcept
            {
                return [&]<const host::label_t... Is>(std::index_sequence<Is...>)
                {
                    return thread::array<T, N>{
                        (data_[size_constant<Is>{}] * A)...};
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
             * @brief Division operator
             * @return The dot product of the first array and the inverse of the second, both of which are of the same type and size
             **/
            __device__ __host__ [[nodiscard]] inline constexpr thread::array<T, N> operator/(const T &A) const __restrict__ noexcept
            {
                return [&]<const host::label_t... Is>(std::index_sequence<Is...>)
                {
                    return thread::array<T, N>{
                        (data_[size_constant<Is>{}] / A)...};
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

            /**
             * @brief Sums all elements in the array.
             * @return Total of every entry in the array.
             **/
            __device__ __host__ [[nodiscard]] inline constexpr T sum() const __restrict__ noexcept
            {
                return [&]<host::label_t... Is>(std::index_sequence<Is...>)
                {
                    return (T{0} + ... + data_[size_constant<Is>{}]);
                }(std::make_index_sequence<N>{});
            }

            /**
             * @brief Computes the number of elements equal to a value in an array
             * @tparam val The value to compare against
             * @tparam Equal Count elements equal t (if true) or not equal to (if false)
             * @return Number of elements in the array equal to val
             **/
            template <const T val, const bool Equal>
            __device__ __host__ [[nodiscard]] inline constexpr host::label_t count() const noexcept
            {
                host::label_t n = 0;

                for (host::label_t i = 0; i < N; i++)
                {
                    if constexpr (Equal)
                    {
                        if (data_[i] == val)
                        {
                            n++;
                        }
                    }
                    else
                    {
                        if (!(data_[i] == val))
                        {
                            n++;
                        }
                    }
                }

                return n;
            }

            /**
             * @brief Returns the index of the K-th element matching the given criterion.
             *
             * @tparam val Value to compare against.
             * @tparam Equal If true, match values equal to val; otherwise match values not equal to val.
             * @tparam K Zero-based index of the matching element to return.
             * @return Index of the K-th matching element.
             * @note This helper is used internally by `indices_of()` and `values_of()`.
             **/
            template <const T val, const bool Equal, const host::label_t K>
            __device__ __host__ constexpr host::label_t get_kth_matching_index() const noexcept
            {
                host::label_t count = 0;
                for (host::label_t i = 0; i < N; ++i)
                {
                    const bool match = Equal ? (data_[i] == val) : !(data_[i] == val);
                    if (match)
                    {
                        if (count == K)
                            return i;
                        ++count;
                    }
                }
                return 0; // fallback
            }

            /**
             * @brief Builds an array of indices whose elements satisfy a comparison criterion.
             *
             * @tparam val Value to compare against.
             * @tparam Equal If true, select elements equal to val; otherwise select elements not equal to val.
             * @tparam ReturnSize Number of matching indices to return.
             * @return Array containing the matching indices.
             **/
            template <const T val, const bool Equal, const host::label_t ReturnSize>
            __device__ __host__ [[nodiscard]] inline constexpr thread::array<host::label_t, ReturnSize> indices_of() const noexcept
            {
                return [&]<host::label_t... Ks>(std::index_sequence<Ks...>)
                {
                    return thread::array<host::label_t, ReturnSize>{get_kth_matching_index<val, Equal, Ks>()...};
                }(std::make_index_sequence<ReturnSize>{});
            }

            /**
             * @brief Builds an array of values whose elements satisfy a comparison criterion.
             *
             * @tparam val Value to compare against.
             * @tparam Equal If true, select elements equal to val; otherwise select elements not equal to val.
             * @tparam ReturnSize Number of matching values to return.
             * @return Array containing the matching values.
             **/
            template <const T val, const bool Equal, const host::label_t ReturnSize>
            __device__ __host__ [[nodiscard]] inline constexpr thread::array<T, ReturnSize> values_of() const noexcept
            {
                return [&]<host::label_t... Ks>(std::index_sequence<Ks...>)
                {
                    return thread::array<T, ReturnSize>{get_kth_matching_value<val, Equal, Ks>()...};
                }(std::make_index_sequence<ReturnSize>{});
            }

            /**
             * @brief Checks if the array contains a specific value
             * @tparam val The value to check for
             * @return True if the array contains val, false otherwise
             **/
            template <const T val>
            __device__ __host__ [[nodiscard]] inline constexpr bool contains() const noexcept
            {
                for (host::label_t i = 0; i < N; i++)
                {
                    if (data_[i] == val)
                    {
                        return true;
                    }
                }
                return false;
            }

            /**
             * @brief Computes the number of non-zero elements of an array
             * @return Number of non-zero elements in the array
             **/
            __device__ __host__ [[nodiscard]] inline constexpr host::label_t number_non_zero() const noexcept
            {
                return count<0, false>();
            }

            /**
             * @brief Get the non-zero values in the array
             * @tparam ReturnSize Size of the returned array
             * @return Array containing only non-zero values from the input array
             **/
            template <const host::label_t ReturnSize>
            __device__ __host__ [[nodiscard]] inline constexpr thread::array<T, ReturnSize> non_zero_values() const noexcept
            {
                return values_of<0, false, ReturnSize>();
            }

            /**
             * @brief Get the non-zero indices in the array
             * @tparam ReturnSize Size of the returned array
             * @return Array containing only non-zero indices from the input array
             **/
            template <const device::label_t ReturnSize>
            __device__ __host__ [[nodiscard]] inline constexpr thread::array<host::label_t, ReturnSize> non_zero_indices() const noexcept
            {
                return indices_of<0, false, ReturnSize>();
            }

        private:
            /**
             * @brief The underlying data
             **/
            T data_[N];

            /**
             * @brief Compile-time check that accesses are valid
             * @tparam i Index of the element
             **/
            template <const host::label_t i>
            __device__ __host__ static inline consteval void assert_legal_access() noexcept
            {
                static_assert(in_bounds<i, N>, "index is out of range: Must be < N.");
            }

            /**
             * @brief Returns the value at the K-th matching position of a comparison criterion.
             *
             * @tparam val Value to compare against.
             * @tparam Equal If true, match values equal to val; otherwise match values not equal to val.
             * @tparam K Ordinal index of the matching element to select.
             * @return The value stored at the selected matching index.
             * @note This is used as the value-producing counterpart to `get_kth_matching_index()`.
             **/
            template <const T val, bool Equal, host::label_t K>
            __device__ __host__ constexpr T get_kth_matching_value() const noexcept
            {
                host::label_t count = 0;
                for (host::label_t i = 0; i < N; ++i)
                {
                    const bool match = Equal ? (data_[i] == val) : !(data_[i] == val);
                    if (match)
                    {
                        if (count == K)
                            return data_[i];
                        ++count;
                    }
                }
                // Unreachable if ReturnSize is correct; fallback for safety
                return T{};
            }
        };
    }

    /**
     * @brief Creates a zero-initialized thread array of the requested size.
     *
     * @tparam T Element type.
     * @tparam N Number of elements.
     * @return Zero-filled `thread::array<T, N>`.
     **/
    template <typename T, const host::label_t N>
    __device__ __host__ [[nodiscard]] inline consteval const thread::array<T, N> zeros() noexcept
    {
        constexpr const std::integral_constant<T, static_cast<T>(0)> value;
        return thread::array<T, N>(value);
    }

    /**
     * @brief Type alias for thread arrays of scalar_t with given size
     * @note This alias simplifies the declaration of certain arrays in the code
     **/
    using momentsArray = thread::array<scalar_t, NUMBER_MOMENTS<host::label_t>()>;
    using scalar = thread::array<scalar_t, 1>;
    using vector = thread::array<scalar_t, 3>;
    using symmetricTensor = thread::array<scalar_t, 6>;
}

#endif

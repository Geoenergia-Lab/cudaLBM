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
    File containing common definitions and functions for all function objects

Namespace
    LBM::functionObjects

SourceFiles
    functionObjects.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_FUNCTIONOBJECTS_CUH
#define __MBLBM_FUNCTIONOBJECTS_CUH

namespace LBM
{
    namespace functionObjects
    {
        using calculateFunction = std::function<void()>;

        using saveFunction = std::function<void(host::array<host::PINNED, scalar_t> &hostWriteBuffer, const host::label_t)>;

        /**
         * @brief Reads an arbitrary list of pointers from devPtrs
         * @tparam ptrIndices The indices of the pointers to read
         * @param[in] devPtrs The pointers to read from
         * @param[in] idx Spatial index
         * @return The values at location idx
         **/
        template <const host::label_t... ptrIndices>
        __device__ [[nodiscard]] inline constexpr const thread::array<scalar_t, sizeof...(ptrIndices)> read_from_moments(const device::ptrColl_t &devPtrs, const device::label_t idx) noexcept
        {
            return {devPtrs.ptr<ptrIndices>()[idx]...};
        }

        /**
         * @brief Reads all pointers from devPtrs
         * @tparam N Number of pointers to read
         * @param[in] devPtrs The pointers to read from
         * @param[in] idx Spatial index
         * @return The values at location idx
         **/
        template <const host::label_t N>
        __device__ [[nodiscard]] inline constexpr const thread::array<scalar_t, N> read(const device::ptrCollection<N, scalar_t> &devPtrs, const device::label_t idx) noexcept
        {
            return [&]<host::label_t... Is>(std::index_sequence<Is...>)
            {
                return thread::array<scalar_t, N>{
                    devPtrs.template ptr<static_cast<host::label_t>(Is)>()[idx]...};
            }(std::make_index_sequence<N>{});
        }

        /**
         * @brief Saves all results to resultPtrs
         * @tparam N Number of pointers to save
         * @param[in] result The result to save
         * @param[out] resultPtrs The pointers to save to
         * @param[in] idx Spatial index
         **/
        template <const host::label_t N>
        __device__ inline void save(const thread::array<scalar_t, N> &result, const device::ptrCollection<N, scalar_t> resultPtrs, const device::label_t idx) noexcept
        {
            device::constexpr_for<0, N>(
                [&](const auto i)
                {
                    resultPtrs.template ptr<i>()[idx] = result[q_i<i>()];
                });
        }

        /**
         * @brief Computes the updated time average of a single scalar value.
         * @tparam T Return type
         * @param[in] fMean Current time average.
         * @param[in] f Current instantaneous value.
         * @param[in] invNewCount Reciprocal of (timeSteps + 1).
         * @return The updated time average.
         **/
        template <typename T>
        __device__ [[nodiscard]] inline constexpr T time_average(const T fMean, const T f, const T invNewCount) noexcept
        {
            return fMean + (f - fMean) * invNewCount;
        }

        /**
         * @brief Helper that applies the scalar time_average element‑wise across an array using an index sequence.
         * @tparam T Element type.
         * @tparam N Array size.
         * @tparam Is Index sequence (deduced internally, not to be called directly).
         * @param[in] fMean Current time‑averaged array.
         * @param[in] f Current instantaneous array.
         * @param[in] invNewCount Reciprocal of (timeSteps + 1).
         * @return Array where each element is the updated time average of the corresponding elements.
         **/
        template <typename T, const host::label_t N, const host::label_t... Is>
        __device__ [[nodiscard]] inline constexpr const thread::array<T, N> time_average(const thread::array<T, N> &fMean, const thread::array<T, N> &f, const T invNewCount, const std::index_sequence<Is...>) noexcept
        {
            return {time_average(fMean[Is], f[Is], invNewCount)...};
        }

        /**
         * @brief Calculates the time average of an array.
         * @tparam T Element type.
         * @tparam N Array size.
         * @param[in] fMean Current time average array.
         * @param[in] f Current instantaneous array.
         * @param[in] invNewCount Reciprocal of (timeSteps + 1).
         * @return The updated time average array.
         **/
        template <typename T, const host::label_t N>
        __device__ [[nodiscard]] inline constexpr const thread::array<T, N> time_average(const thread::array<T, N> &fMean, const thread::array<T, N> &f, const T invNewCount) noexcept
        {
            return time_average(fMean, f, invNewCount, std::make_index_sequence<N>{});
        }

        /**
         * @brief Computes the squared difference between two scalars: (a - b) ^ 2.
         * @tparam T Return type
         * @param[in] a First value.
         * @param[in] b Second value.
         * @return (a - b) * (a - b).
         **/
        template <typename T>
        __device__ [[nodiscard]] inline constexpr T squared_difference(const T a, const T b) noexcept
        {
            return (a - b) * (a - b);
        }

        /**
         * @brief Helper that applies the scalar squared_difference element‑wise across two arrays using an index sequence.
         * @tparam T Element type.
         * @tparam N Array size.
         * @tparam Is Index sequence (deduced internally, not to be called directly).
         * @param[in] a First array.
         * @param[in] b Second array.
         * @return Array where each element is (a[i] - b[i]) ^ 2.
         **/
        template <typename T, const host::label_t N, const host::label_t... Is>
        __device__ [[nodiscard]] inline constexpr const thread::array<T, N> squared_difference(const thread::array<T, N> &a, const thread::array<T, N> &b, const std::index_sequence<Is...>) noexcept
        {
            return {squared_difference(a[Is], b[Is])...};
        }

        /**
         * @brief Calculates the element‑wise squared difference between two arrays: (a[i] - b[i]) ^ 2
         * @tparam T Element type.
         * @tparam N Array size.
         * @param[in] a First array.
         * @param[in] b Second array.
         * @return Array of squared differences.
         **/
        template <typename T, const host::label_t N>
        __device__ [[nodiscard]] inline constexpr const thread::array<T, N> squared_difference(const thread::array<T, N> &a, const thread::array<T, N> &b) noexcept
        {
            return squared_difference(a, b, std::make_index_sequence<N>{});
        }

        /**
         * @brief Device-side function for calculating the time averaged quantity only
         * @tparam FunctionObject The function object to calculate
         * @param[in] devPtrs Device pointer collection containing density, velocity and moment fields
         * @param[out] resultMeanPtrs Device pointer collection for the time averaged quantity
         * @param[in] invNewCount Reciprocal of (nTimeSteps + 1) for time averaging
         **/
        template <class FunctionObject>
        __device__ inline void mean(
            const device::ptrColl_t &devPtrs,
            const device::ptrCollection<FunctionObject::N, scalar_t> &resultMeanPtrs,
            const scalar_t invNewCount) noexcept
        {
            // Calculate the index
            const device::label_t idx = device::idx(thread::coordinate(), block::coordinate());

            // Calculate the instantaneous
            const thread::array<scalar_t, FunctionObject::N> resultInstantaneous = FunctionObject::calculate(devPtrs, idx);

            // Read the mean values from global memory
            const thread::array<scalar_t, FunctionObject::N> resultMean = read(resultMeanPtrs, idx);

            // Update the mean value and write back to global
            const thread::array<scalar_t, FunctionObject::N> resultMeanNew = time_average(resultMean, resultInstantaneous, invNewCount);

            save(resultMeanNew, resultMeanPtrs, idx);
        }

        /**
         * @brief Device-side function for calculating the instantaneous and time averaged quantity
         * @tparam FunctionObject The function object to calculate
         * @param[in] devPtrs Device pointer collection containing density, velocity and moment fields
         * @param[out] resultPtrs Device pointer collection for the instantaneous quantity
         * @param[out] resultMeanPtrs Device pointer collection for the time averaged quantity
         * @param[in] invNewCount Reciprocal of (nTimeSteps + 1) for time averaging
         **/
        template <class FunctionObject>
        __device__ inline void instantaneousAndMean(
            const device::ptrColl_t &devPtrs,
            const device::ptrCollection<FunctionObject::N, scalar_t> &resultPtrs,
            const device::ptrCollection<FunctionObject::N, scalar_t> &resultMeanPtrs,
            const scalar_t invNewCount) noexcept
        {
            // Calculate the index
            const device::label_t idx = device::idx(thread::coordinate(), block::coordinate());

            // Calculate the instantaneous
            const thread::array<scalar_t, FunctionObject::N> resultInstantaneous = FunctionObject::calculate(devPtrs, idx);

            // Save the instantaneous to global memory
            save(resultInstantaneous, resultPtrs, idx);

            // Read the mean values from global memory
            const thread::array<scalar_t, FunctionObject::N> resultMean = read(resultMeanPtrs, idx);

            // Update the mean value
            const thread::array<scalar_t, FunctionObject::N> resultMeanNew = time_average(resultMean, resultInstantaneous, invNewCount);

            // Write the mean value back to global
            save(resultMeanNew, resultMeanPtrs, idx);
        }

        /**
         * @brief Device-side function for calculating the instantaneous quantity only
         * @tparam FunctionObject The function object to calculate
         * @param[in] devPtrs Device pointer collection containing density, velocity and moment fields
         * @param[out] resultPtrs Device pointer collection for the instantaneous quantity
         **/
        template <class FunctionObject>
        __device__ inline void instantaneous(
            const device::ptrColl_t &devPtrs,
            const device::ptrCollection<FunctionObject::N, scalar_t> &resultPtrs) noexcept
        {
            // Calculate the index
            const device::label_t idx = device::idx(thread::coordinate(), block::coordinate());

            // Calculate the instantaneous
            const thread::array<scalar_t, FunctionObject::N> resultInstantaneous = FunctionObject::calculate(devPtrs, idx);

            // Save the instantaneous to global memory
            save(resultInstantaneous, resultPtrs, idx);
        }

        /**
         * @brief Device-side function for calculating the instantaneous quantity only
         * @tparam FunctionObject The function object to calculate
         * @param[in] devPtrs Device pointer collection containing density, velocity and moment fields
         * @param[in] resultMeanPtrs Device pointer collection for the time averaged quantity
         * @param[out] resultPrimePtrs Device pointer collection for the instantaneous quantity
         **/
        template <class FunctionObject>
        __device__ inline void prime(
            const device::ptrColl_t &devPtrs,
            const device::ptrCollection<FunctionObject::N, scalar_t> &resultMeanPtrs,
            const device::ptrCollection<FunctionObject::N, scalar_t> &resultPrimePtrs) noexcept
        {
            // Calculate the index
            const device::label_t idx = device::idx(thread::coordinate(), block::coordinate());

            // Calculate the instantaneous
            const thread::array<scalar_t, FunctionObject::N> resultInstantaneous = FunctionObject::calculate(devPtrs, idx);

            // Read the mean values from global memory
            const thread::array<scalar_t, FunctionObject::N> resultMean = read(resultMeanPtrs, idx);

            // Update the prime value
            const thread::array<scalar_t, FunctionObject::N> resultPrimeNew = resultInstantaneous - resultMean;

            // Write the prime value back to global
            save(resultPrimeNew, resultPrimePtrs, idx);
        }

        /**
         * @brief Device-side function for calculating the time average of the perturbation quantity
         * @tparam FunctionObject The function object to calculate
         * @param[in] devPtrs Device pointer collection containing density, velocity and moment fields
         * @param[out] resultMeanPtrs Device pointer collection for the time averaged quantity
         * @param[out] resultPrimeSqMeanPtrs Device pointer collection for the time averaged quantity
         * @param[in] invNewCount Reciprocal of (nTimeSteps + 1) for time averaging
         **/
        template <class FunctionObject>
        __device__ inline void primeSqMean(
            const device::ptrColl_t &devPtrs,
            const device::ptrCollection<FunctionObject::N, scalar_t> &resultMeanPtrs,
            const device::ptrCollection<FunctionObject::N, scalar_t> &resultPrimeSqMeanPtrs,
            const scalar_t invNewCount) noexcept
        {
            // Calculate the index
            const device::label_t idx = device::idx(thread::coordinate(), block::coordinate());

            // Calculate the instantaneous
            const thread::array<scalar_t, FunctionObject::N> resultInstantaneous = FunctionObject::calculate(devPtrs, idx);

            // Read the mean values from global memory
            const thread::array<scalar_t, FunctionObject::N> resultMean = read(resultMeanPtrs, idx);

            // Read the prime mean value
            const thread::array<scalar_t, FunctionObject::N> resultPrimeSqMean = read(resultPrimeSqMeanPtrs, idx);

            // Update the prime squared value
            const thread::array<scalar_t, FunctionObject::N> resultPrimeSqNew = squared_difference(resultInstantaneous, resultMean);

            // Update the prime squared mean value
            const thread::array<scalar_t, FunctionObject::N> resultPrimeSqMeanNew = time_average(resultPrimeSqMean, resultPrimeSqNew, invNewCount);

            // Write the prime mean value back to global
            save(resultPrimeSqMeanNew, resultPrimeSqMeanPtrs, idx);
        }

        /**
         * @brief Initializes calculation switches based on function object configuration
         * @param[in] objectName Name of the function object to check
         * @return True if the object is enabled in configuration
         **/
        __host__ [[nodiscard]] bool initialiserSwitch(const name_t &objectName) noexcept
        {
            return std::filesystem::exists("functionObjects") ? string::containsString(string::trim<true>(string::eraseBraces(string::extractBlock(string::readFile("functionObjects"), "functionObjectList"))), objectName) : false;
        }
    }
}

#endif
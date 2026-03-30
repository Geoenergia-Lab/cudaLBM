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
        using saveFunction = std::function<void(const host::label_t)>;

        /**
         * @brief The names of the 10 solution variables of the moment representation
         **/
        const words_t solutionVariableNames{"rho", "u", "v", "w", "m_xx", "m_xy", "m_xz", "m_yy", "m_yz", "m_zz"};

        /**
         * @brief Maps the name of function objects to lists of the names of their individual components
         **/
        const std::unordered_map<name_t, words_t> fieldComponentsMap = {
            {"momentsMean", {"rhoMean", "uMean", "vMean", "wMean", "m_xxMean", "m_xyMean", "m_xzMean", "m_yyMean", "m_yzMean", "m_zzMean"}},
            {"rho", {"rho"}},
            {"U", {"U_x", "U_y", "U_z"}},
            {"Pi", {"Pi_xx", "Pi_xy", "Pi_xz", "Pi_yy", "Pi_yz", "Pi_zz"}},
            {"S", {"S_xx", "S_xy", "S_xz", "S_yy", "S_yz", "S_zz"}},
            {"SMean", {"S_xxMean", "S_xyMean", "S_xzMean", "S_yyMean", "S_yzMean", "S_zzMean"}},
            {"k", {"k"}},
            {"kMean", {"kMean"}}};

        /**
         * @brief Reads an arbitrary list of pointers from devPtrs
         * @tparam ptrIndices The indices of the pointers to read
         * @param[in] devPtrs The pointers to read from
         * @param[in] idx Spatial index
         * @return The values at location idx
         **/
        template <const host::label_t... ptrIndices>
        __device__ [[nodiscard]] inline constexpr const thread::array<scalar_t, sizeof...(ptrIndices)> read_from_moments(
            const device::ptrCollection<NUMBER_MOMENTS<host::label_t>(), const scalar_t> &devPtrs,
            const device::label_t idx) noexcept
        {
            return {devPtrs.ptr<ptrIndices>()[idx]...};
        }

        /**
         * @brief Reads all pointers from devPtrs
         * @param[in] devPtrs The pointers to read from
         * @param[in] idx Spatial index
         * @return The values at location idx
         **/
        template <const host::label_t N>
        __device__ [[nodiscard]] inline constexpr const thread::array<scalar_t, N> read(
            const device::ptrCollection<N, scalar_t> &devPtrs,
            device::label_t idx) noexcept
        {
            thread::array<scalar_t, N> result;

            device::constexpr_for<0, N>(
                [&](const auto i)
                {
                    result[i] = devPtrs.template ptr<i>()[idx];
                });

            return result;
        }

        /**
         * @brief Saves all results to resultPtrs
         * @param[in] result The result to save
         * @param[out] resultPtrs The pointers to save to
         * @param[in] idx Spatial index
         **/
        template <const host::label_t N>
        __device__ inline void save(
            const thread::array<scalar_t, N> &result,
            const device::ptrCollection<N, scalar_t> resultPtrs,
            const device::label_t idx) noexcept
        {
            device::constexpr_for<0, N>(
                [&](const auto i)
                {
                    resultPtrs.template ptr<i>()[idx] = result[q_i<i>()];
                });
        }

        /**
         * @brief Calculates the time average of a variable
         * @param[in] fMean The current time average value
         * @param[in] f The current instantaneous value
         * @param[in] invNewCount The reciprocal of (nTimeSteps + 1)
         * @return The updated time average
         **/
        template <typename T>
        __device__ [[nodiscard]] inline constexpr T timeAverage(const T fMean, const T f, const T invNewCount) noexcept
        {
            return fMean + (f - fMean) * invNewCount;
        }

        /**
         * @brief Calculates the time average of an array
         * @param[in] fMean The current time average value
         * @param[in] f The current instantaneous value
         * @param[in] invNewCount The reciprocal of (nTimeSteps + 1)
         * @return The updated time average
         **/
        template <typename T, const host::label_t N>
        __device__ [[nodiscard]] inline constexpr const thread::array<T, N> timeAverage(
            const thread::array<T, N> &fMean,
            const thread::array<T, N> &f,
            const T invNewCount) noexcept
        {
            thread::array<T, N> newMean;

            device::constexpr_for<0, N>(
                [&](const auto n)
                {
                    newMean[n] = timeAverage(fMean[n], f[n], invNewCount);
                });

            return newMean;
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
            const device::ptrCollection<NUMBER_MOMENTS<host::label_t>(), const scalar_t> &devPtrs,
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
            const thread::array<scalar_t, FunctionObject::N> resultMeanNew = timeAverage(resultMean, resultInstantaneous, invNewCount);

            save(resultMeanNew, resultMeanPtrs, idx);
        }

        /**
         * @brief Device-side function for calculating the instantaneous and time averaged quantity
         * @tparam FunctionObject The function object to calculate
         * @param[in] devPtrs Device pointer collection containing density, velocity and moment fields
         * @param[out] resulPtrs Device pointer collection for the instantaneous quantity
         * @param[out] resultMeanPtrs Device pointer collection for the time averaged quantity
         * @param[in] invNewCount Reciprocal of (nTimeSteps + 1) for time averaging
         **/
        template <class FunctionObject>
        __device__ inline void instantaneousAndMean(
            const device::ptrCollection<NUMBER_MOMENTS<host::label_t>(), const scalar_t> &devPtrs,
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
            const thread::array<scalar_t, FunctionObject::N> resultMeanNew = timeAverage(resultMean, resultInstantaneous, invNewCount);

            // Write the mean value back to global
            save(resultMeanNew, resultMeanPtrs, idx);
        }

        /**
         * @brief Device-side function for calculating the instantaneous quantity only
         * @tparam FunctionObject The function object to calculate
         * @param[in] devPtrs Device pointer collection containing density, velocity and moment fields
         * @param[out] resulPtrs Device pointer collection for the instantaneous quantity
         **/
        template <class FunctionObject>
        __device__ inline void instantaneous(
            const device::ptrCollection<NUMBER_MOMENTS<host::label_t>(), const scalar_t> &devPtrs,
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
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
        /**
         * @brief The names of the 10 solution variables of the moment representation
         **/
        const std::vector<std::string> solutionVariableNames{"rho", "u", "v", "w", "m_xx", "m_xy", "m_xz", "m_yy", "m_yz", "m_zz"};

        /**
         * @brief Maps the name of function objects to lists of the names of their individual components
         **/
        const std::unordered_map<std::string, std::vector<std::string>> fieldComponentsMap = {
            {"momentsMean", {"rhoMean", "uMean", "vMean", "wMean", "m_xxMean", "m_xyMean", "m_xzMean", "m_yyMean", "m_yzMean", "m_zzMean"}},
            {"S", {"S_xx", "S_xy", "S_xz", "S_yy", "S_yz", "S_zz"}},
            {"SMean", {"S_xxMean", "S_xyMean", "S_xzMean", "S_yyMean", "S_yzMean", "S_zzMean"}},
            {"k", {"k"}},
            {"kMean", {"kMean"}}};

        /**
         * @brief Calculates the time average of a variable
         * @return The updated time average
         * @param[in] fMean The current time average value
         * @param[in] f The current instantaneous value
         * @param[in] invNewCount The reciprocal of (nTimeSteps + 1)
         **/
        template <typename T>
        __device__ [[nodiscard]] inline constexpr T timeAverage(const T fMean, const T f, const T invNewCount) noexcept
        {
            return fMean + (f - fMean) * invNewCount;
        }

        template <typename T, const label_t N>
        __device__ [[nodiscard]] inline constexpr const thread::array<T, N> timeAverage(const thread::array<T, N> &fMean, const thread::array<T, N> &f, const T invNewCount) noexcept
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
         * @brief Initializes calculation switches based on function object configuration
         * @param[in] objectName Name of the function object to check
         * @return True if the object is enabled in configuration
         **/
        __host__ [[nodiscard]] bool initialiserSwitch(const std::string &objectName)
        {
            return std::filesystem::exists("functionObjects") ? string::containsString(string::trim<true>(string::eraseBraces(string::extractBlock(string::readFile("functionObjects"), "functionObjectList"))), objectName) : false;
        }

        /**
         * @brief Allocates either a regular or zero-initialized device array based on the allocate flag
         * @return A device::array either allocated to a uniform value or non-allocated
         * @param[in] name The name of the variable to construct
         * @param[in] mesh The lattice mesh
         * @param[in] allocate Determines whether or not to allocate the variable
         **/
        template <class VelocitySet, const time::type TimeType>
        __host__ [[nodiscard]] device::array<field::FULL_FIELD, scalar_t, VelocitySet, TimeType> objectAllocator(
            const std::string &name,
            const host::latticeMesh &mesh)
        {
            // If we wish to allocate the array, do so
            if (initialiserSwitch(name))
            {
                return device::array<field::FULL_FIELD, scalar_t, VelocitySet, TimeType>(name, mesh, 0);
            }
            // Otherwise, just create the array without initializing it
            else
            {
                return device::array<field::FULL_FIELD, scalar_t, VelocitySet, TimeType>(name, mesh);
            }
        }

        template <class VelocitySet, const time::type TimeType>
        __host__ [[nodiscard]] device::array<field::FULL_FIELD, scalar_t, VelocitySet, TimeType> objectAllocator(
            const std::string &name,
            const host::latticeMesh &mesh,
            const bool allocate)
        {
            // If we wish to allocate the array, do so
            if (allocate)
            {
                return device::array<field::FULL_FIELD, scalar_t, VelocitySet, TimeType>(name, mesh, 0);
            }
            // Otherwise, just create the array without initializing it
            else
            {
                return device::array<field::FULL_FIELD, scalar_t, VelocitySet, TimeType>(name, mesh);
            }
        }
    }
}

#endif
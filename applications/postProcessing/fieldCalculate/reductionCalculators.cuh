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
    Defines calculation operations involving reduction operations

Namespace
    LBM

SourceFiles
    reductionCalculators.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_REDUCTIONCALCULATORS_CUH
#define __MBLBM_REDUCTIONCALCULATORS_CUH

namespace LBM
{
    namespace calculator
    {
        /**
         * @brief Transforms a value according to the sign policy.
         *
         * @tparam Sign absMode::ABS to return the absolute value, absMode::SIGNED to return unchanged.
         * @tparam T Numeric type.
         * @param[in] val Input value.
         * @return std::abs(val) if Sign == ABS, else val.
         **/
        template <const numericalSchemes::absMode Sign, typename T>
        __host__ [[nodiscard]] inline constexpr T fix_sign(const T val) noexcept
        {
            if constexpr (Sign == numericalSchemes::ABS)
            {
                return std::abs(val);
            }
            else
            {
                return val;
            }
        }

        /**
         * @brief Computes the extremum (max or min) of a single field vector.
         * @tparam Sign absMode policy for signed/absolute values.
         * @tparam T Field element type.
         * @tparam Compare Comparison functor (e.g., std::less for min, std::greater for max).
         * @param[in] field The 1D field data.
         * @param[in] comp Comparator defining the extremum.
         * @return The extremum value (raw or absolute, as specified by Sign).
         **/
        template <const numericalSchemes::absMode Sign, typename T, typename Compare>
        __host__ [[nodiscard]] T fieldExtremaImpl(const std::vector<T> &field, const Compare comp) noexcept
        {
            T extremum = fix_sign<Sign>(field[0]);

            for (const T &value : field)
            {
                const T fixed_value = fix_sign<Sign>(value);
                if (comp(fixed_value, extremum))
                {
                    extremum = fixed_value;
                }
            }
            return extremum;
        }

        /**
         * @brief Calculates the spatial mean of a field
         * @tparam ReturnType The return type
         * @tparam T Type of the variable to sum
         * @param[in] field The field to calculate the mean of
         * @return The spatial mean of the field
         **/
        template <typename ReturnType, typename T>
        __host__ [[nodiscard]] inline ReturnType spatialSum(const std::vector<T> &field) noexcept
        {
            double sum = static_cast<double>(0);
            for (const T &value : field)
            {
                sum += static_cast<double>(value);
            }
            return static_cast<ReturnType>(sum);
        }

        /**
         * @brief Calculates the spatial mean of a field
         * @tparam T Type of the variable to sum
         * @param[in] field The field to calculate the mean of
         * @return The spatial mean of the field
         **/
        template <typename T>
        __host__ [[nodiscard]] inline T spatialMean(const std::vector<T> &field) noexcept
        {
            return static_cast<T>(spatialSum<double>(field) / static_cast<double>(field.size()));
        }

        /**
         * @brief Checks if a field contains any NaN values
         * @tparam T Type of the field to check
         * @param[in] field The field to check
         * @param[out] status Status of the calculation (1 if field contains NaN)
         * @return True if the field contains NaN values, false otherwise
         **/
        template <typename T>
        __host__ [[nodiscard]] inline host::label_t containsNaN(const std::vector<T> &field, int &status) noexcept
        {
            host::label_t count = 0;
            for (const T &value : field)
            {
                if (std::isnan(value))
                {
                    status = 1;
                    count++;
                }
            }
            return count;
        }

        /**
         * @brief Prints a per‑field reduction for all fields in a collection.
         * @tparam Reducer A callable that computes a scalar from a field vector, e.g. fieldExtremaImpl<Sign> or spatialMean.
         * @param[in] variables The arrayCollection of fields.
         * @param[in] mesh Reference to the lattice mesh
         * @param[in] timeStep Current time step.
         * @param[in] compute The reduction function. Must have signature scalar_t(const std::vector<scalar_t>&) or be a template that can accept a single argument.
         * @param[in] label A string printed before the value (used as-is).
         **/
        template <const bool Deinterleave, const bool Sort, typename Reducer>
        __host__ void printFieldReduction(
            const host::arrayCollection<scalar_t> &variables,
            const host::latticeMesh &mesh,
            const host::label_t timeStep,
            const Reducer &compute,
            const std::string &label) noexcept
        {
            const std::vector<std::vector<scalar_t>> fields = variables.splitFields<Deinterleave, Sort>(mesh);
            std::cout << "Time: " << timeStep << std::endl;
            std::cout << "{" << std::endl;
            for (host::label_t field = 0; field < fields.size(); field++)
            {
                std::cout << std::setprecision(15) << IO::whitespace<4>{} << label << "(" << variables.varNames()[field] << "): " << compute(fields[field]) << ";" << std::endl;
            }
            std::cout << "};" << std::endl;
        }

        /**
         * @brief Prints a per‑field reduction for all fields in a collection.
         * @tparam Reducer A callable that computes a scalar from a field vector, e.g. fieldExtremaImpl<Sign> or spatialMean.
         * @param[in] variables The arrayCollection of fields.
         * @param[in] mesh Reference to the lattice mesh
         * @param[in] timeStep Current time step.
         * @param[in] comp Comparator (e.g., std::greater for max)
         * @param[in] label A string printed before the value (used as-is).
         **/
        template <const numericalSchemes::absMode Sign, typename Compare>
        __host__ void fieldExtrema(
            const host::arrayCollection<scalar_t> &variables,
            const host::latticeMesh &mesh,
            const host::label_t timeStep,
            const Compare comp,
            const std::string &label) noexcept
        {
            const auto compute = [comp](const std::vector<scalar_t> &f)
            {
                return fieldExtremaImpl<Sign>(f, comp);
            };
            printFieldReduction<false, false>(variables, mesh, timeStep, compute, label);
        }

        /**
         * @brief Convenience function - prints the maximum of each field (signed values).
         **/
        __host__ void fieldMax(
            const host::arrayCollection<scalar_t> &variables,
            const host::latticeMesh &mesh,
            const host::label_t timeStep,
            [[maybe_unused]] int &status,
            [[maybe_unused]] const name_t &fileName) noexcept
        {
            fieldExtrema<numericalSchemes::SIGNED>(variables, mesh, timeStep, std::greater<scalar_t>(), "max");
        }

        /**
         * @brief Convenience function - prints the minimum of each field (signed values).
         **/
        __host__ void fieldMin(
            const host::arrayCollection<scalar_t> &variables,
            const host::latticeMesh &mesh,
            const host::label_t timeStep,
            [[maybe_unused]] int &status,
            [[maybe_unused]] const name_t &fileName) noexcept
        {
            fieldExtrema<numericalSchemes::SIGNED>(variables, mesh, timeStep, std::less<scalar_t>(), "min");
        }

        /**
         * @brief Convenience function - prints the absolute maximum of each field.
         **/
        __host__ void fieldAbsMax(
            const host::arrayCollection<scalar_t> &variables,
            const host::latticeMesh &mesh,
            const host::label_t timeStep,
            [[maybe_unused]] int &status,
            [[maybe_unused]] const name_t &fileName) noexcept
        {
            fieldExtrema<numericalSchemes::ABS>(variables, mesh, timeStep, std::greater<scalar_t>(), "maxAbs");
        }

        /**
         * @brief Convenience function - prints the absolute minimum of each field.
         **/
        __host__ void fieldAbsMin(
            const host::arrayCollection<scalar_t> &variables,
            const host::latticeMesh &mesh,
            const host::label_t timeStep,
            [[maybe_unused]] int &status,
            [[maybe_unused]] const name_t &fileName) noexcept
        {
            fieldExtrema<numericalSchemes::ABS>(variables, mesh, timeStep, std::less<scalar_t>(), "minAbs");
        }

        /**
         * @brief Convenience function - prints the spatial mean of each field
         **/
        __host__ void spatialMean(
            const host::arrayCollection<scalar_t> &variables,
            const host::latticeMesh &mesh,
            const host::label_t timeStep,
            [[maybe_unused]] int &status,
            [[maybe_unused]] const name_t &fileName) noexcept
        {
            const auto compute = [](const std::vector<scalar_t> &f)
            {
                return spatialMean<scalar_t>(f);
            };

            printFieldReduction<false, true>(variables, mesh, timeStep, compute, "mean");
        }

        /**
         * @brief Convenience function - prints the spatial sum of each field
         **/
        __host__ void spatialSum(
            const host::arrayCollection<scalar_t> &variables,
            const host::latticeMesh &mesh,
            const host::label_t timeStep,
            [[maybe_unused]] int &status,
            [[maybe_unused]] const name_t &fileName) noexcept
        {
            const auto compute = [](const std::vector<scalar_t> &f)
            {
                return spatialSum<scalar_t>(f);
            };

            printFieldReduction<false, true>(variables, mesh, timeStep, compute, "sum");
        }

        /**
         * @brief Convenience function - prints the spatial sum of each field
         **/
        __host__ void containsNaN(
            const host::arrayCollection<scalar_t> &variables,
            const host::latticeMesh &mesh,
            const host::label_t timeStep,
            int &status,
            [[maybe_unused]] const name_t &fileName) noexcept
        {
            const auto compute = [&](const std::vector<scalar_t> &f)
            {
                return containsNaN(f, status);
            };

            printFieldReduction<false, false>(variables, mesh, timeStep, compute, "containsNaN");
        }
    }
}

#endif
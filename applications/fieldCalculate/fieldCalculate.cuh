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
    Function definitions and includes specific to the fieldCalculate executable

Namespace
    LBM

SourceFiles
    fieldCalculate.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_FIELDCALCULATE_CUH
#define __MBLBM_FIELDCALCULATE_CUH

#include "../../src/LBMIncludes.cuh"
#include "../../src/typedefs/typedefs.cuh"
#include "../../src/array/threadArray.cuh"
#include "../../src/collision/collision.cuh"
#include "../../src/blockHalo/blockHalo.cuh"
#include "../../src/fileIO/fileIO.cuh"
#include "../../src/runTimeIO/runTimeIO.cuh"
#include "../../src/postProcess/postProcess.cuh"
#include "../../src/programControl/programControl.cuh"
#include "../../src/numericalSchemes/numericalSchemes.cuh"
#include "../fieldConvert/fieldConvert.cuh"

namespace LBM
{
    __host__ [[nodiscard]] inline consteval label_t SchemeOrder() { return 8; }

    /**
     * @brief Checks if a field contains any NaN values
     * @param[in] field The field to check
     * @return True if the field contains NaN values, false otherwise
     **/
    __host__ [[nodiscard]] inline bool containsNaN(const std::vector<scalar_t> &field) noexcept
    {
        for (const auto &value : field)
        {
            if (std::isnan(value))
            {
                return true;
            }
        }
        return false;
    }

    /**
     * @brief Checks if any of the fields in the arrayCollection contain NaN values and prints the result
     * @param[in] variables The arrayCollection containing the fields to check
     * @param[in] mesh The lattice mesh
     * @param[in] timeStep The current time step for logging purposes
     **/
    __host__ void containsNaN(
        const host::arrayCollection<scalar_t, ctorType::MUST_READ> &variables,
        const host::latticeMesh &mesh,
        const label_t timeStep) noexcept
    {
        // De-interleave the fields
        const std::vector<std::vector<scalar_t>> fields = fileIO::deinterleaveAoS(variables.arr(), mesh);

        std::cout << "Time: " << timeStep << std::endl;
        std::cout << "{" << std::endl;

        std::size_t numberNaNs = 0;

        // Loop over the fields checking for NaN
        for (std::size_t field = 0; field < fields.size(); field++)
        {
            if (containsNaN(fields[field]))
            {
                std::cout << "    NaN detected in field " << variables.varNames()[field] << std::endl;
                numberNaNs++;
            }
        }

        if (numberNaNs == 0)
        {
            std::cout << "    No NaN values detected in any field." << std::endl;
        }
        else
        {
            std::cout << "    Total number of fields with NaN values: " << numberNaNs << std::endl;
        }

        std::cout << "};" << std::endl;

        return;
    }

    /**
     * @brief Calculates the spatial mean of a field
     * @param[in] field The field to calculate the mean of
     * @return The spatial mean of the field
     **/
    __host__ [[nodiscard]] inline scalar_t spatialMean(const std::vector<scalar_t> &field) noexcept
    {
        scalar_t sum = static_cast<scalar_t>(0);
        for (const auto &value : field)
        {
            sum += value;
        }
        return sum / static_cast<scalar_t>(field.size());
    }

    /**
     * @brief Calculates and prints the spatial mean of each field in the arrayCollection
     * @param[in] variables The arrayCollection containing the fields to calculate the mean of
     * @param[in] mesh The lattice mesh
     * @param[in] timeStep The current time step for logging purposes
     **/
    __host__ void spatialMean(
        const host::arrayCollection<scalar_t, ctorType::MUST_READ> &variables,
        const host::latticeMesh &mesh,
        const label_t timeStep) noexcept
    {
        // De-interleave the fields
        const std::vector<std::vector<scalar_t>> fields = fileIO::deinterleaveAoS(variables.arr(), mesh);

        std::cout << "Time: " << timeStep << std::endl;
        std::cout << "{" << std::endl;

        for (std::size_t field = 0; field < fields.size(); field++)
        {
            const scalar_t fieldMean = spatialMean(fields[field]);
            std::cout << "    mean(" << variables.varNames()[field] << "): " << fieldMean << ";" << std::endl;
        }

        std::cout << "};" << std::endl;
    }
}

#endif
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
    Function definitions and includes specific to the fieldConvert executable

Namespace
    LBM

SourceFiles
    fieldConvert.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_FIELDCONVERT_CUH
#define __MBLBM_FIELDCONVERT_CUH

#include "../../src/LBMIncludes.cuh"
#include "../../src/LBMTypedefs.cuh"
#include "../../src/strings.cuh"
#include "../../src/array/array.cuh"
#include "../../src/collision/collision.cuh"
#include "../../src/blockHalo/blockHalo.cuh"
#include "../../src/fileIO/fileIO.cuh"
#include "../../src/runTimeIO/runTimeIO.cuh"
#include "../../src/postProcess/postProcess.cuh"
#include "../../src/inputControl.cuh"
#include "../../src/functionObjects/functionObjects.cuh"

namespace LBM
{
    /**
     * @brief Creates an error message for invalid writer types
     * @param[in] writerNames Unordered map of the writer types to the appropriate functions
     * @param[in] conversion The invalid conversion type provided by the user
     * @return A formatted error message listing the supported formats
     **/
    __host__ [[nodiscard]] const std::string invalidWriter(const std::unordered_map<std::string, postProcess::writerFunction> &writerNames, const std::string &conversion) noexcept
    {
        std::vector<std::string> supportedFormats;
        for (const auto &pair : writerNames)
        {
            supportedFormats.push_back(pair.first);
        }

        // Sort them alphabetically
        std::sort(supportedFormats.begin(), supportedFormats.end());

        // Create the error message with supported formats
        std::string errorMsg = "Unsupported conversion format: " + conversion + "\nSupported formats are: ";
        for (std::size_t i = 0; i < supportedFormats.size(); ++i)
        {
            if (i != 0)
            {
                errorMsg += ", ";
            }
            errorMsg += supportedFormats[i];
        }

        return errorMsg;
    }

    /**
     * @brief Returns the field names based on the provided prefix and whether a custom field is specified
     * @param[in] fileNamePrefix The prefix for the field names
     * @param[in] doCustomField Boolean indicating if a custom field is specified
     * @return A reference to a vector of field names
     * @throws std::runtime_error if an invalid field name is provided
     **/
    __host__ [[nodiscard]] host::arrayCollection<scalar_t, ctorType::MUST_READ> initialiseArrays(
        const std::string &fileNamePrefix,
        const programControl &programCtrl,
        const std::vector<std::string> &fieldNames,
        const label_t timeStep)
    {
        // Construct from a custom field name
        if (programCtrl.input().isArgPresent("-fieldName"))
        {
            return host::arrayCollection<scalar_t, ctorType::MUST_READ>(fileNamePrefix, fieldNames, timeStep);
        }
        // Otherwise construct from default field names
        else
        {
            return host::arrayCollection<scalar_t, ctorType::MUST_READ>(programCtrl, fieldNames, timeStep);
        }
    }

    /**
     * @brief Returns the field names based on the provided prefix and whether a custom field is specified
     * @param[in] fileNamePrefix The prefix for the field names
     * @param[in] doCustomField Boolean indicating if a custom field is specified
     * @return A reference to a vector of field names
     * @throws std::runtime_error if an invalid field name is provided
     **/
    __host__ [[nodiscard]] const std::vector<std::string> &getFieldNames(
        const std::string &fileNamePrefix,
        const bool doCustomField)
    {
        if (!doCustomField)
        {
            return functionObjects::solutionVariableNames;
        }
        else
        {
            const std::unordered_map<std::string, std::vector<std::string>>::const_iterator namesIterator = functionObjects::fieldComponentsMap.find(fileNamePrefix);
            const bool foundField = namesIterator != functionObjects::fieldComponentsMap.end();
            if (!foundField)
            {
                // Throw an exception: invalid field name
                throw std::runtime_error("Invalid argument passed to -fieldName");
            }
            else
            {
                return namesIterator->second;
            }
        }
    }

    namespace direction
    {
        typedef enum Enum : label_t
        {
            x = 0,
            y = 1,
            z = 2,
            UNDEFINED = 3
        } cardinal;
    }

    __host__ [[nodiscard]] direction::cardinal cutPlaneDirection(const programControl &programCtrl) noexcept
    {
        const std::string cutPlanePrefix = programCtrl.getArgument("-cutPlane");

        // Need to check that j = 1 because the first character before the = symbol should be x, y or z and nothing else
        if (!(string::findCharPosition(cutPlanePrefix, "=") == 1))
        {
            return direction::UNDEFINED;
        }

        if (cutPlanePrefix[0] == "x"[0])
        {
            return direction::x;
        }

        if (cutPlanePrefix[0] == "y"[0])
        {
            return direction::y;
        }

        if (cutPlanePrefix[0] == "z"[0])
        {
            return direction::z;
        }

        return direction::UNDEFINED;
    }

    __host__ [[nodiscard]] inline host::latticeMesh meshSlice(const host::latticeMesh &mesh, const direction::cardinal dir) noexcept
    {
        if (dir == direction::x)
        {
            return host::latticeMesh(mesh, {1, mesh.ny(), mesh.nz()});
        }

        if (dir == direction::y)
        {
            return host::latticeMesh(mesh, {mesh.nx(), 1, mesh.nz()});
        }

        if (dir == direction::z)
        {
            return host::latticeMesh(mesh, {mesh.nx(), mesh.ny(), 1});
        }

        return host::latticeMesh(mesh, {1, 1, 1});
    }

    template <const direction::cardinal dir>
    __host__ [[nodiscard]] std::vector<std::vector<scalar_t>> initialiseSlice(
        const host::latticeMesh &mesh,
        const label_t nFields)
    {
        static_assert((dir == direction::x) | (dir == direction::y) | (dir == direction::z), "Bad direction");

        if constexpr (dir == direction::x)
        {
            return std::vector<std::vector<scalar_t>>(nFields, std::vector<scalar_t>(mesh.ny() * mesh.nz(), 0));
        }

        if constexpr (dir == direction::y)
        {
            return std::vector<std::vector<scalar_t>>(nFields, std::vector<scalar_t>(mesh.nx() * mesh.nz(), 0));
        }

        if constexpr (dir == direction::z)
        {
            return std::vector<std::vector<scalar_t>>(nFields, std::vector<scalar_t>(mesh.nx() * mesh.ny(), 0));
        }
    }

    template <const direction::cardinal dir>
    __host__ [[nodiscard]] scalar_t indexCoordinate(const host::latticeMesh &mesh, const scalar_t pointCoordinate)
    {
        static_assert((dir == direction::x) | (dir == direction::y) | (dir == direction::z), "Bad direction");

        if constexpr (dir == direction::x)
        {
            return static_cast<scalar_t>(mesh.nx() - 1) * (pointCoordinate * mesh.L().x);
        }
        if constexpr (dir == direction::y)
        {
            return static_cast<scalar_t>(mesh.ny() - 1) * (pointCoordinate * mesh.L().y);
        }
        if constexpr (dir == direction::z)
        {
            return static_cast<scalar_t>(mesh.nz() - 1) * (pointCoordinate * mesh.L().z);
        }
    }

    template <typename T>
    __host__ [[nodiscard]] T linearInterpolate(const T f0, const T f1, const T weight) noexcept
    {
        return ((static_cast<T>(1) - weight) * f0) + (weight * f1);
    }

    template <const direction::cardinal dir>
    __host__ [[nodiscard]] const std::vector<std::vector<scalar_t>> extractCutPlane(
        const std::vector<std::vector<scalar_t>> &fields,
        const host::latticeMesh &mesh,
        const scalar_t pointCoordinate)
    {
        static_assert((dir == direction::x) | (dir == direction::y) | (dir == direction::z), "Bad direction");

        // Get the "index" coordinate
        const scalar_t i = indexCoordinate<dir>(mesh, pointCoordinate);
        const label_t i_0 = static_cast<label_t>(std::floor(i));
        const label_t i_1 = static_cast<label_t>(std::ceil(i));
        const scalar_t weight = pointCoordinate - static_cast<scalar_t>(i_0);

        std::vector<std::vector<scalar_t>> cutPlane = initialiseSlice<dir>(mesh, fields.size());

        if constexpr (dir == direction::x)
        {
            for (std::size_t field = 0; field < fields.size(); field++)
            {
                for (std::size_t z = 0; z < mesh.nz(); z++)
                {
                    for (std::size_t y = 0; y < mesh.ny(); y++)
                    {
                        const scalar_t f0 = fields[field][host::idxScalarGlobal(i_0, y, z, mesh.nx(), mesh.ny())];
                        const scalar_t f1 = fields[field][host::idxScalarGlobal(i_1, y, z, mesh.nx(), mesh.ny())];

                        cutPlane[field][y + (z * mesh.ny())] = linearInterpolate(f0, f1, weight);
                    }
                }
            }
            return cutPlane;
        }
        if constexpr (dir == direction::y)
        {
            for (std::size_t field = 0; field < fields.size(); field++)
            {
                for (std::size_t z = 0; z < mesh.nz(); z++)
                {
                    for (std::size_t x = 0; x < mesh.nx(); x++)
                    {
                        const scalar_t f0 = fields[field][host::idxScalarGlobal(x, i_0, z, mesh.nx(), mesh.ny())];
                        const scalar_t f1 = fields[field][host::idxScalarGlobal(x, i_1, z, mesh.nx(), mesh.ny())];
                        cutPlane[field][x + (z * mesh.nx())] = linearInterpolate(f0, f1, weight);
                    }
                }
            }
            return cutPlane;
        }
        if constexpr (dir == direction::z)
        {
            for (std::size_t field = 0; field < fields.size(); field++)
            {
                for (std::size_t y = 0; y < mesh.ny(); y++)
                {
                    for (std::size_t x = 0; x < mesh.nx(); x++)
                    {
                        const scalar_t f0 = fields[field][host::idxScalarGlobal(x, y, i_0, mesh.nx(), mesh.ny())];
                        const scalar_t f1 = fields[field][host::idxScalarGlobal(x, y, i_1, mesh.nx(), mesh.ny())];
                        cutPlane[field][x + (y * mesh.nx())] = linearInterpolate(f0, f1, weight);
                    }
                }
            }
            return cutPlane;
        }
    }

    __host__ [[nodiscard]] const std::vector<std::vector<scalar_t>> extractCutPlane(
        const std::vector<std::vector<scalar_t>> &fields,
        const host::latticeMesh &mesh,
        const direction::cardinal dir,
        const scalar_t pointCoordinate)
    {
        switch (dir)
        {
        case direction::x:
        {
            return extractCutPlane<direction::x>(fields, mesh, pointCoordinate);
        }
        case direction::y:
        {
            return extractCutPlane<direction::y>(fields, mesh, pointCoordinate);
        }
        case direction::z:
        {
            return extractCutPlane<direction::z>(fields, mesh, pointCoordinate);
        }
        default:
        {
            throw std::runtime_error("Invalid cardinal direction");
        }
        }
    }

    __host__ [[nodiscard]] const std::vector<std::vector<scalar_t>> processFields(
        const host::arrayCollection<scalar_t, ctorType::MUST_READ> &hostMoments,
        const host::latticeMesh &mesh,
        const programControl &programCtrl,
        const bool doCutPlane)
    {
        if (doCutPlane)
        {
            const std::string cutPlanePrefix = programCtrl.getArgument("-cutPlane");

            // Check that size() - 1 isn't = 2
            const scalar_t planeCoordinate = static_cast<scalar_t>(std::stold(cutPlanePrefix.substr(2, cutPlanePrefix.size() - 1)));

            const direction::cardinal dir = cutPlaneDirection(programCtrl);

            return extractCutPlane(
                fileIO::deinterleaveAoS(hostMoments.arr(), mesh),
                mesh,
                dir,
                planeCoordinate);
        }
        else
        {
            return fileIO::deinterleaveAoS(hostMoments.arr(), mesh);
        }
    }

    __host__ [[nodiscard]] const host::latticeMesh processMesh(
        const host::latticeMesh &mesh,
        const programControl &programCtrl,
        const bool cutPlane)
    {
        if (cutPlane)
        {
            const direction::cardinal dir = cutPlaneDirection(programCtrl);

            return meshSlice(mesh, dir);
        }
        else
        {
            return mesh;
        }
    }

    __host__ [[nodiscard]] const std::string processName(const programControl programCtrl, const std::string &fileNamePrefix, const label_t nameIndex, const bool cutPlane)
    {
        // Get the file name at the present time step
        if (cutPlane)
        {
            const std::string cutPlanePrefix = programCtrl.getArgument("-cutPlane");

            return fileNamePrefix + "CutPlane_" + cutPlanePrefix + "_" + std::to_string(nameIndex);
        }
        else
        {
            return fileNamePrefix + "_" + std::to_string(nameIndex);
        }
    }
}

#endif
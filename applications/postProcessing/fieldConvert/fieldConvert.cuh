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

#include "../../../src/LBMIncludes.cuh"
#include "../../../src/typedefs/typedefs.cuh"
#include "../../../src/strings.cuh"
#include "../../../src/array/array.cuh"
#include "../../../src/collision/collision.cuh"
#include "../../../src/blockHalo/blockHalo.cuh"
#include "../../../src/fileIO/fileIO.cuh"
#include "../../../src/runTimeIO/runTimeIO.cuh"
#include "../../../src/postProcess/postProcess.cuh"
#include "../../../src/programControl/programControl.cuh"
#include "../../../src/functionObjects/functionObjects.cuh"

namespace LBM
{
    /**
     * @brief Creates an error message for invalid writer types
     * @param[in] writerNames Unordered map of the writer types to the appropriate functions
     * @param[in] conversion The invalid conversion type provided by the user
     * @return A formatted error message listing the supported formats
     **/
    __host__ [[nodiscard]] const name_t invalidWriter(const std::unordered_map<name_t, postProcess::writerFunction> &writerNames, const name_t &conversion) noexcept
    {
        words_t supportedFormats;
        for (const auto &pair : writerNames)
        {
            supportedFormats.push_back(pair.first);
        }

        // Sort them alphabetically
        std::sort(supportedFormats.begin(), supportedFormats.end());

        // Create the error message with supported formats
        name_t errorMsg = "Unsupported conversion format: " + conversion + "\nSupported formats are: ";
        for (host::label_t i = 0; i < supportedFormats.size(); ++i)
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
    __host__ [[nodiscard]] inline host::arrayCollection<scalar_t, ctorType::MUST_READ> initialiseArrays(
        const name_t &fileNamePrefix,
        const programControl &programCtrl,
        const words_t &fieldNames,
        const host::label_t timeStep)
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
    __host__ [[nodiscard]] inline const words_t &getFieldNames(
        const name_t &fileNamePrefix,
        const bool doCustomField)
    {
        if (!doCustomField)
        {
            return functionObjects::solutionVariableNames;
        }
        else
        {
            const std::unordered_map<name_t, words_t>::const_iterator namesIterator = functionObjects::fieldComponentsMap.find(fileNamePrefix);
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

    __host__ [[nodiscard]] axis::type cutPlaneDirection(const programControl &programCtrl) noexcept
    {
        const name_t cutPlanePrefix = programCtrl.getArgument("-cutPlane");

        // Need to check that j = 1 because the first character before the = symbol should be x, y or z and nothing else
        if (!(string::findCharPosition(cutPlanePrefix, "=") == 1))
        {
            return axis::NO_DIRECTION;
        }

        if (cutPlanePrefix[0] == "x"[0])
        {
            return axis::X;
        }

        if (cutPlanePrefix[0] == "y"[0])
        {
            return axis::Y;
        }

        if (cutPlanePrefix[0] == "z"[0])
        {
            return axis::Z;
        }

        return axis::NO_DIRECTION;
    }

    __host__ [[nodiscard]] inline host::latticeMesh meshSlice(const host::latticeMesh &mesh, const axis::type alpha) noexcept
    {
        if (alpha == axis::X)
        {
            return host::latticeMesh(mesh, {1, mesh.dimension<axis::Y>(), mesh.dimension<axis::Z>()});
        }

        if (alpha == axis::Y)
        {
            return host::latticeMesh(mesh, {mesh.dimension<axis::X>(), 1, mesh.dimension<axis::Z>()});
        }

        if (alpha == axis::Z)
        {
            return host::latticeMesh(mesh, {mesh.dimension<axis::X>(), mesh.dimension<axis::Y>(), 1});
        }

        return host::latticeMesh(mesh, {1, 1, 1});
    }

    template <const axis::type alpha>
    __host__ [[nodiscard]] inline constexpr std::vector<std::vector<scalar_t>> initialiseSlice(
        const host::latticeMesh &mesh,
        const host::label_t nFields)
    {
        axis::assertions::validate<alpha, axis::NOT_NULL>();

        return std::vector<std::vector<scalar_t>>(nFields, std::vector<scalar_t>(mesh.dimension<axis::orthogonal<alpha, 0>()>() * mesh.dimension<axis::orthogonal<alpha, 1>()>(), 0));
    }

    template <const axis::type alpha>
    __host__ [[nodiscard]] inline constexpr scalar_t indexCoordinate(const host::latticeMesh &mesh, const scalar_t pointCoordinate)
    {
        axis::assertions::validate<alpha, axis::NOT_NULL>();

        return (static_cast<scalar_t>(mesh.dimension<alpha>()) * (pointCoordinate / mesh.L().value<alpha>())) - static_cast<scalar_t>(1);
    }

    template <typename T>
    __host__ [[nodiscard]] inline constexpr T linearInterpolate(const T f0, const T f1, const T weight) noexcept
    {
        return ((static_cast<T>(1) - weight) * f0) + (weight * f1);
    }

    template <const axis::type alpha>
    __host__ [[nodiscard]] const std::vector<std::vector<scalar_t>> extractCutPlane(
        const std::vector<std::vector<scalar_t>> &fields,
        const host::latticeMesh &mesh,
        const scalar_t pointCoordinate)
    {
        axis::assertions::validate<alpha, axis::NOT_NULL>();

        // Get the "index" coordinate
        const scalar_t index = indexCoordinate<alpha>(mesh, pointCoordinate);
        const host::label_t index_0 = static_cast<host::label_t>(std::floor(index));
        const host::label_t index_1 = static_cast<host::label_t>(std::ceil(index));
        const scalar_t weight = pointCoordinate - static_cast<scalar_t>(index_0);

        std::vector<std::vector<scalar_t>> cutPlane = initialiseSlice<alpha>(mesh, fields.size());

        // If the points are coincident, no need to interpolate
        if (index_0 == index_1)
        {
            for (host::label_t field = 0; field < fields.size(); field++)
            {
                global::forAllInPlane<alpha>(
                    mesh.dimensions(),
                    [&](const host::label_t i, const host::label_t j)
                    {
                        const host::blockLabel Tx = axis::to_3d<alpha>(i, j, index_0);

                        const host::label_t idx = global::idx(Tx.x, Tx.y, Tx.z, mesh.dimension<axis::X>(), mesh.dimension<axis::Y>());

                        const host::label_t id = i + (j * mesh.dimension<axis ::orthogonal<alpha, 0>()>());

                        cutPlane[field][id] = fields[field][idx];
                    });
            }
        }
        // Otherwise we will need to interpolate between the two points
        else
        {
            for (host::label_t field = 0; field < fields.size(); field++)
            {
                global::forAllInPlane<alpha>(
                    mesh.dimensions(),
                    [&](const host::label_t i, const host::label_t j)
                    {
                        const host::blockLabel Tx_0 = axis::to_3d<alpha>(i, j, index_0);
                        const host::blockLabel Tx_1 = axis::to_3d<alpha>(i, j, index_1);

                        const host::label_t idx_0 = global::idx(Tx_0.x, Tx_0.y, Tx_0.z, mesh.dimension<axis::X>(), mesh.dimension<axis::Y>());
                        const host::label_t idx_1 = global::idx(Tx_1.x, Tx_1.y, Tx_1.z, mesh.dimension<axis::X>(), mesh.dimension<axis::Y>());

                        const scalar_t f0 = fields[field][idx_0];
                        const scalar_t f1 = fields[field][idx_1];

                        const host::label_t id = i + (j * mesh.dimension<axis ::orthogonal<alpha, 0>()>());

                        cutPlane[field][id] = linearInterpolate(f0, f1, weight);
                    });
            }
        }

        return cutPlane;
    }

    __host__ [[nodiscard]] inline constexpr const std::vector<std::vector<scalar_t>> extractCutPlane(
        const std::vector<std::vector<scalar_t>> &fields,
        const host::latticeMesh &mesh,
        const axis::type alpha,
        const scalar_t pointCoordinate)
    {
        switch (alpha)
        {
        case axis::X:
        {
            return extractCutPlane<axis::X>(fields, mesh, pointCoordinate);
        }
        case axis::Y:
        {
            return extractCutPlane<axis::Y>(fields, mesh, pointCoordinate);
        }
        case axis::Z:
        {
            return extractCutPlane<axis::Z>(fields, mesh, pointCoordinate);
        }
        default:
        {
            throw std::runtime_error("Invalid cardinal direction");
        }
        }
    }

    __host__ [[nodiscard]] inline const std::vector<std::vector<scalar_t>> processFields(
        const host::arrayCollection<scalar_t, ctorType::MUST_READ> &hostMoments,
        const host::latticeMesh &mesh,
        const programControl &programCtrl,
        const bool doCutPlane)
    {
        if (doCutPlane)
        {
            const name_t cutPlanePrefix = programCtrl.getArgument("-cutPlane");

            // Check that size() - 1 isn't = 2
            std::cout << "Doing cut plane at " << cutPlanePrefix.substr(2, cutPlanePrefix.size() - 1) << std::endl;
            const scalar_t planeCoordinate = static_cast<scalar_t>(std::stold(cutPlanePrefix.substr(2, cutPlanePrefix.size() - 1)));

            const axis::type alpha = cutPlaneDirection(programCtrl);

            return extractCutPlane(
                fileIO::deinterleaveAoS(hostMoments.arr(), mesh),
                mesh,
                alpha,
                planeCoordinate);
        }
        else
        {
            return fileIO::deinterleaveAoS(hostMoments.arr(), mesh);
        }
    }

    __host__ [[nodiscard]] inline const host::latticeMesh processMesh(
        const host::latticeMesh &mesh,
        const programControl &programCtrl,
        const bool cutPlane)
    {
        if (cutPlane)
        {
            const axis::type alpha = cutPlaneDirection(programCtrl);

            return meshSlice(mesh, alpha);
        }
        else
        {
            return host::latticeMesh(programCtrl);
        }
    }

    __host__ [[nodiscard]] inline const name_t processName(const programControl &programCtrl, const name_t &fileNamePrefix, const host::label_t nameIndex, const bool cutPlane)
    {
        // Get the file name at the present time step
        if (cutPlane)
        {
            const name_t cutPlanePrefix = programCtrl.getArgument("-cutPlane");

            return fileNamePrefix + "CutPlane_" + cutPlanePrefix + "_" + std::to_string(nameIndex);
        }
        else
        {
            return fileNamePrefix + "_" + std::to_string(nameIndex);
        }
    }
}

#endif
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
    Function definitions and includes specific to the fieldConvert executable

Namespace
    LBM

SourceFiles
    fieldConvert.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_FIELDCONVERT_CUH
#define __MBLBM_FIELDCONVERT_CUH

#include "../postProcessingIncludes.cuh"

namespace LBM
{
    __host__ [[nodiscard]] axis::type cutPlaneDirection(const programControl &programCtrl) noexcept
    {
        const name_t cutPlanePrefix = programCtrl.getArgument("-cutPlane");

        // Need to check that j = 1 because the first character before the = symbol should be x, y or z and nothing else
        if (!(string::findCharPosition<"="[0]>(cutPlanePrefix) == 1))
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

        return (static_cast<scalar_t>(mesh.dimension<alpha>() - static_cast<host::label_t>(1)) * (pointCoordinate / mesh.L().value<alpha>())) - static_cast<scalar_t>(1);
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
                        const host::pointLabel Tx = axis::to_3d<alpha>(i, j, index_0);

                        const host::label_t idx = Cartesian::idx(Tx.x, Tx.y, Tx.z, mesh.dimension<axis::X>(), mesh.dimension<axis::Y>());

                        const host::label_t id = i + (j * mesh.dimension<axis::orthogonal<alpha, 0>()>());

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
                        const host::pointLabel Tx_0 = axis::to_3d<alpha>(i, j, index_0);
                        const host::pointLabel Tx_1 = axis::to_3d<alpha>(i, j, index_1);

                        const host::label_t idx_0 = Cartesian::idx(Tx_0.x, Tx_0.y, Tx_0.z, mesh.dimension<axis::X>(), mesh.dimension<axis::Y>());
                        const host::label_t idx_1 = Cartesian::idx(Tx_1.x, Tx_1.y, Tx_1.z, mesh.dimension<axis::X>(), mesh.dimension<axis::Y>());

                        const scalar_t f0 = fields[field][idx_0];
                        const scalar_t f1 = fields[field][idx_1];

                        const host::label_t id = i + (j * mesh.dimension<axis::orthogonal<alpha, 0>()>());

                        cutPlane[field][id] = numericalSchemes::interpolate<scalar_t>::linear(f0, f1, weight);
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

    __host__ [[nodiscard]] const std::vector<std::vector<scalar_t>> processFields(
        const host::arrayCollection<scalar_t> &hostMoments,
        const host::latticeMesh &mesh,
        const programControl &programCtrl,
        const bool doCutPlane)
    {
        if (doCutPlane)
        {
            const name_t cutPlanePrefix = programCtrl.getArgument("-cutPlane");

            // Check that size() - 1 isn't = 2
            const scalar_t planeCoordinate = static_cast<scalar_t>(std::stold(cutPlanePrefix.substr(2, cutPlanePrefix.size() - 1)));

            const axis::type alpha = cutPlaneDirection(programCtrl);

            return extractCutPlane(
                hostMoments.deinterleaveAoS(mesh),
                mesh,
                alpha,
                planeCoordinate);
        }
        else
        {
            return hostMoments.deinterleaveAoS(mesh);
        }
    }

    __host__ [[nodiscard]] const name_t processName(const programControl &programCtrl, const name_t &fileNamePrefix, const host::label_t nameIndex, const bool cutPlane)
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
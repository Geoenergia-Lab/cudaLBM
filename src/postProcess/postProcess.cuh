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
    Top-level header file for the post processing routines

Namespace
    LBM::postProcess

SourceFiles
    postProcess.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_POSTPROCESS_CUH
#define __MBLBM_POSTPROCESS_CUH

#include "../LBMIncludes.cuh"
#include "../typedefs/typedefs.cuh"
#include "../fileIO/fileIO.cuh"

namespace LBM
{
    namespace postProcess
    {
        /**
         * @brief Builds the physical coordinate array for each lattice node.
         *
         * @tparam T Coordinate value type.
         * @param[in] mesh Lattice mesh defining the domain geometry.
         * @return Vector of physical coordinates in interleaved form:
         *         [x0, y0, z0, x1, y1, z1, ...].
         *
         * @details Converts each lattice index into a physical coordinate using the mesh
         * spacing and dimensions. The values are stored in a single flat vector.
         **/
        template <typename T>
        __host__ [[nodiscard]] const std::vector<T> meshCoordinates(const host::latticeMesh &mesh)
        {
            std::vector<T> coords(mesh.size() * 3, 0);

            global::forAll(
                mesh.dimensions(),
                host::blockLabel(0, 0, 0),
                [&](const host::label_t x, const host::label_t y, const host::label_t z)
                {
                    const host::label_t idx = Cartesian::idx(x, y, z, mesh.dimension<axis::X>(), mesh.dimension<axis::Y>());
                    // Do the conversion in double, then cast to the desired type
                    coords[3 * idx + 0] = static_cast<T>((static_cast<double>(mesh.L().x) * static_cast<double>(x * static_cast<host::label_t>(mesh.dimension<axis::X>() > 1))) / static_cast<double>(mesh.dimension<axis::X>() - static_cast<host::label_t>(mesh.dimension<axis::X>() > 1)));
                    coords[3 * idx + 1] = static_cast<T>((static_cast<double>(mesh.L().y) * static_cast<double>(y * static_cast<host::label_t>(mesh.dimension<axis::Y>() > 1))) / static_cast<double>(mesh.dimension<axis::Y>() - static_cast<host::label_t>(mesh.dimension<axis::Y>() > 1)));
                    coords[3 * idx + 2] = static_cast<T>((static_cast<double>(mesh.L().z) * static_cast<double>(z * static_cast<host::label_t>(mesh.dimension<axis::Z>() > 1))) / static_cast<double>(mesh.dimension<axis::Z>() - static_cast<host::label_t>(mesh.dimension<axis::Z>() > 1)));
                });

            return coords;
        }

        /**
         * @brief Builds the connectivity list for each hexahedral cell in the mesh.
         *
         * @tparam one_based If true, element indices are one-based; otherwise zero-based.
         * @tparam IndexType Integer type used for the connectivity entries.
         * @param[in] mesh Lattice mesh to process.
         * @return Vector containing the cell connectivity for the mesh.
         *
         * @details Each cell contributes eight point indices forming a hexahedron.
         **/
        template <const bool one_based, typename IndexType>
        __host__ [[nodiscard]] const std::vector<IndexType> meshConnectivity(const host::latticeMesh &mesh)
        {
            const host::label_t nx = mesh.dimension<axis::X>();
            const host::label_t ny = mesh.dimension<axis::Y>();
            const host::label_t nz = mesh.dimension<axis::Z>();
            const host::label_t numElements = (nx - 1) * (ny - 1) * (nz - 1);

            std::vector<IndexType> connectivity(numElements * 8, 0);
            constexpr const device::label_t offset = one_based ? 1 : 0;
            global::forAll(
                host::blockLabel(nx - 1, ny - 1, nz - 1),
                host::blockLabel(0, 0, 0),
                [&](const host::label_t x, const host::label_t y, const host::label_t z)
                {
                    const host::label_t base = Cartesian::idx(x, y, z, mesh.dimension<axis::X>(), mesh.dimension<axis::Y>());
                    const host::label_t cell_idx = Cartesian::idx(x, y, z, mesh.dimension<axis::X>() - 1, mesh.dimension<axis::Y>() - 1);
                    const host::label_t stride_y = mesh.dimension<axis::X>();
                    const host::label_t stride_z = mesh.dimension<axis::X>() * mesh.dimension<axis::Y>();

                    connectivity[cell_idx * 8 + 0] = static_cast<IndexType>(base + offset);
                    connectivity[cell_idx * 8 + 1] = static_cast<IndexType>(base + 1 + offset);
                    connectivity[cell_idx * 8 + 2] = static_cast<IndexType>(base + stride_y + 1 + offset);
                    connectivity[cell_idx * 8 + 3] = static_cast<IndexType>(base + stride_y + offset);
                    connectivity[cell_idx * 8 + 4] = static_cast<IndexType>(base + stride_z + offset);
                    connectivity[cell_idx * 8 + 5] = static_cast<IndexType>(base + stride_z + 1 + offset);
                    connectivity[cell_idx * 8 + 6] = static_cast<IndexType>(base + stride_z + stride_y + 1 + offset);
                    connectivity[cell_idx * 8 + 7] = static_cast<IndexType>(base + stride_z + stride_y + offset);
                });

            return connectivity;
        }

        /**
         * @brief Builds the cell offset array for the mesh.
         *
         * @tparam T Integer type used for the offsets.
         * @param[in] mesh Lattice mesh.
         * @return Vector containing the per-cell offset values.
         *
         * @details The returned values are the cumulative offsets used for VTK-like
         * connectivity or element writing.
         **/
        template <typename T>
        __host__ [[nodiscard]] const std::vector<T> meshOffsets(const host::latticeMesh &mesh)
        {
            const host::label_t nx = mesh.dimension<axis::X>();
            const host::label_t ny = mesh.dimension<axis::Y>();
            const host::label_t nz = mesh.dimension<axis::Z>();
            const host::label_t numElements = (nx - 1) * (ny - 1) * (nz - 1);

            std::vector<T> offsets(numElements);

            for (host::label_t i = 0; i < numElements; ++i)
            {
                offsets[i] = static_cast<T>((i + 1) * 8);
            }

            return offsets;
        }
    }

    class writer
    {
    public:
        static constexpr const char *directoryPrefix = "postProcess";

        /**
         * @brief Verifies that the output directory has enough free space for a file export.
         *
         * @tparam Writer Output file writer type.
         * @param[in] mesh Lattice mesh used to estimate output size.
         * @param[in] varNames Variable names included in the exported field data.
         * @param[in] fileName Base file name for the output.
         *
         * @details Computes the expected disk usage from the mesh and variable metadata
         * and throws if the file cannot be written safely.
         **/
        template <class Writer>
        __host__ static inline void diskSpaceAssertion(const host::latticeMesh &mesh, const words_t &varNames, const name_t &fileName)
        {
            fileSystem::diskSpaceAssertion<
                Writer::file_format,
                Writer::has_fields,
                Writer::has_points,
                Writer::has_elements,
                Writer::has_offsets>(
                mesh,
                varNames.size(),
                fileName);
        }

        /**
         * @brief Prints a status entry to the console.
         *
         * @param[in] key Status label.
         * @param[in] value Boolean result to display.
         **/
        __host__ static inline void printStatus(const name_t &key, const bool value) noexcept
        {
            std::cout << IO::whitespace<4>{} << key << ": " << (value ? "OK;" : "Fail;") << std::endl;
        }

        /**
         * @brief Writes post-processed solution data to the requested output format.
         *
         * @tparam Writer Concrete output writer type.
         * @param[in] solutionVars Solution variables to export, grouped by field.
         * @param[in] fileName Output file name without extension.
         * @param[in] mesh Mesh describing the geometry.
         * @param[in] varNames Names of the exported solution variables.
         *
         * @details Validates the variable count and field sizes, creates the output
         * directory if needed, checks disk space, and writes the file using the selected
         * writer backend.
         **/
        template <class Writer>
        __host__ static void write(
            const std::vector<std::vector<scalar_t>> &solutionVars,
            const name_t &fileName,
            const host::latticeMesh &mesh,
            const words_t &varNames)
        {
            const host::label_t numNodes = mesh.dimension<axis::X>() * mesh.dimension<axis::Y>() * mesh.dimension<axis::Z>();
            const host::label_t numVars = solutionVars.size();

            if (numVars != varNames.size())
            {
                throw std::runtime_error("Error: The number of solution (" + std::to_string(numVars) + ") does not match the count of variable names (" + std::to_string(varNames.size()) + ")");
            }

            for (host::label_t i = 0; i < numVars; i++)
            {
                if (solutionVars[i].size() != numNodes)
                {
                    throw std::runtime_error("Error: The solution variable " + std::to_string(i) + " has " + std::to_string(solutionVars[i].size()) + " elements, expected " + std::to_string(numNodes));
                }
            }

            const name_t trueFileName(name_t(directoryPrefix) + "/" + fileName + Writer::fileExtension);

            std::cout << Writer::name << std::endl;
            std::cout << "{" << std::endl;
            std::cout << IO::whitespace<4>{} << "fileName: " << trueFileName << ";" << std::endl;

            const bool directoryStatus = fileSystem::makeDirectory(directoryPrefix);

            printStatus("directory", directoryStatus);

            std::cout << IO::whitespace<4>{} << "fileSize: " << fileSystem::to_MiB<double>(fileSystem::expectedDiskUsage<Writer::file_format, Writer::has_fields, Writer::has_points, Writer::has_elements, Writer::has_offsets>(mesh, solutionVars.size())) << " MiB;" << std::endl;

            // Check if there is enough disk space to store the file
            writer::diskSpaceAssertion<Writer>(mesh, varNames, fileName);

            constexpr std::ios::openmode mode = std::ios::out | (Writer::file_format == fileSystem::BINARY ? std::ios::binary : std::ios::openmode(0));
            std::ofstream outFile(trueFileName, mode);

            if (!outFile)
            {
                std::cout << "};" << std::endl;
                throw std::runtime_error("Error opening file: " + trueFileName);
            }

            const bool writeStatus = Writer::write(solutionVars, outFile, mesh, varNames);

            printStatus("ofstream", writeStatus);

            std::cout << "};" << std::endl;
        }
    };
}

#include "Tecplot.cuh"
#include "VTK.cuh"
#include "VTU.cuh"
#include "VTS.cuh"
#include "VTI.cuh"
#include "LBMBin.cuh"
#include "writerFunction.cuh"

#endif
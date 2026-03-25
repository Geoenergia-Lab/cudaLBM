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
         * @brief Calculates physical coordinates of lattice points
         * @tparam T Coordinate data type (typically scalar_t or double)
         * @param[in] mesh The lattice mesh
         * @return Vector of coordinates in interleaved format [x0, y0, z0, x1, y1, z1, ...]
         *
         * This function converts lattice indices to physical coordinates using
         * the domain dimensions stored in the mesh. Coordinates are normalized
         * to the physical domain size and distributed evenly across the lattice.
         **/
        template <typename T>
        __host__ [[nodiscard]] const std::vector<T> meshCoordinates(const host::latticeMesh &mesh)
        {
            const host::label_t nx = mesh.dimension<axis::X>();
            const host::label_t ny = mesh.dimension<axis::Y>();
            const host::label_t nz = mesh.dimension<axis::Z>();
            const host::label_t nPoints = nx * ny * nz;

            std::vector<T> coords(nPoints * 3, 0);

            global::forAll(
                mesh.dimensions(),
                host::blockLabel(0, 0, 0),
                [&](const host::label_t x, const host::label_t y, const host::label_t z)
                {
                    const host::label_t idx = global::idx(x, y, z, mesh.dimension<axis::X>(), mesh.dimension<axis::Y>());
                    // Do the conversion in double, then cast to the desired type
                    coords[3 * idx + 0] = static_cast<T>((static_cast<double>(mesh.L().x) * static_cast<double>(x * static_cast<host::label_t>(mesh.dimension<axis::X>() > 1))) / static_cast<double>(mesh.dimension<axis::X>() - static_cast<host::label_t>(mesh.dimension<axis::X>() > 1)));
                    coords[3 * idx + 1] = static_cast<T>((static_cast<double>(mesh.L().y) * static_cast<double>(y * static_cast<host::label_t>(mesh.dimension<axis::Y>() > 1))) / static_cast<double>(mesh.dimension<axis::Y>() - static_cast<host::label_t>(mesh.dimension<axis::Y>() > 1)));
                    coords[3 * idx + 2] = static_cast<T>((static_cast<double>(mesh.L().z) * static_cast<double>(z * static_cast<host::label_t>(mesh.dimension<axis::Z>() > 1))) / static_cast<double>(mesh.dimension<axis::Z>() - static_cast<host::label_t>(mesh.dimension<axis::Z>() > 1)));
                });

            return coords;
        }

        /**
         * @brief Calculates the connectivity of the points of a latticeMesh object
         * @tparam one_based If true, indices are 1-based. If false, 0-based.
         * @tparam IndexType The integer type for the connectivity data (e.g., uint32_t, host::label_t).
         * @param[in] mesh The lattice mesh
         * @return An std::vector of type IndexType containing the latticeMesh object connectivity
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
                    const host::label_t base = global::idx(x, y, z, mesh.dimension<axis::X>(), mesh.dimension<axis::Y>());
                    const host::label_t cell_idx = global::idx(x, y, z, mesh.dimension<axis::X>() - 1, mesh.dimension<axis::Y>() - 1);
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
         * @brief Calculates the point offsets of the points of a latticeMesh object
         * @tparam T The integer type for the offset data (e.g., uint32_t, host::label_t).
         * @param[in] mesh The lattice mesh
         * @return An std::vector of type T containing the latticeMesh object point offsets
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

        /**
         * @brief Obtain the name of the type that corresponds to the C++ data type
         * @tparam T The C++ data type (e.g. float, int64_t)
         * @return A string containing the name of the VTK type (e.g. "Float32", "Int64")
         **/
        template <typename T>
        __host__ [[nodiscard]] inline consteval const char *getVtkTypeName() noexcept
        {
            if constexpr (std::is_same_v<T, float>)
            {
                return "Float32";
            }
            else if constexpr (std::is_same_v<T, double>)
            {
                return "Float64";
            }
            else if constexpr (std::is_same_v<T, int32_t>)
            {
                return "Int32";
            }
            else if constexpr (std::is_same_v<T, uint32_t>)
            {
                return "UInt32";
            }
            else if constexpr (std::is_same_v<T, int64_t>)
            {
                return "Int64";
            }
            else if constexpr (std::is_same_v<T, uint64_t>)
            {
                return "UInt64";
            }
            else if constexpr (std::is_same_v<T, uint8_t>)
            {
                return "UInt8";
            }
            else if constexpr (std::is_same_v<T, int8_t>)
            {
                return "Int8";
            }
            else
            {
                static_assert(std::is_same_v<T, void>, "Unsupported type for getVtkTypeName");
                return "Unknown";
            }
        }
    }

    class writer
    {
    public:
        __host__ [[nodiscard]] static inline consteval const char *directoryPrefix() noexcept { return "postProcess"; }

        template <class Writer>
        static inline void diskSpaceAssertion(const host::latticeMesh &mesh, const words_t &varNames, const name_t &fileName)
        {
            fileSystem::diskSpaceAssertion<
                Writer::format(),
                Writer::hasFields(),
                Writer::hasPoints(),
                Writer::hasElements(),
                Writer::hasOffsets()>(
                mesh,
                varNames.size(),
                fileName);
        }

        __host__ static inline void printStatus(const name_t &key, const bool value) noexcept
        {
            std::cout << "    " << key << ": " << (value ? "OK;" : "Fail;") << std::endl;
        }

        /**
         * @brief Templated writer function for post-processing
         * @tparam Writer The type of file output (VTU, VTS, Tecplot)
         * @param[in] solutionVars The solution variables to write
         * @param[in] fileName The name of the file to be written
         * @param[in] mesh The lattice mesh
         * @param[in] varNames The names of the variables to write
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
                throw std::runtime_error("Error: The number of solution (" + std::to_string(numVars) + ") does not match the count of variable names (" + std::to_string(varNames.size()));
            }

            for (host::label_t i = 0; i < numVars; i++)
            {
                if (solutionVars[i].size() != numNodes)
                {
                    throw std::runtime_error("Error: The solution variable " + std::to_string(i) + " has " + std::to_string(solutionVars[i].size()) + " elements, expected " + std::to_string(numNodes));
                }
            }

            const name_t trueFileName(name_t(directoryPrefix()) + "/" + fileName + Writer::fileExtension());

            std::cout << Writer::name() << std::endl;
            std::cout << "{" << std::endl;
            std::cout << "    fileName: " << trueFileName << ";" << std::endl;

            const bool directoryStatus = fileSystem::makeDirectory(directoryPrefix());

            printStatus("directory", directoryStatus);

            std::cout << "    fileSize: " << fileSystem::to_MiB<double>(fileSystem::expectedDiskUsage<Writer::format(), Writer::hasFields(), Writer::hasPoints(), Writer::hasElements(), Writer::hasOffsets()>(mesh, solutionVars.size())) << " MiB;" << std::endl;

            // Check if there is enough disk space to store the file
            writer::diskSpaceAssertion<Writer>(mesh, varNames, fileName);

            std::ofstream outFile(trueFileName);

            if (!outFile)
            {
                std::cout << "};" << std::endl;
                throw std::runtime_error("Error opening file: " + trueFileName);
            }

            const bool writeStatus = Writer::write(solutionVars, outFile, mesh, varNames);

            printStatus("ofstream", outFile.good());

            std::cout << "};" << std::endl;
        }
    };
}

#include "Tecplot.cuh"
#include "VTU.cuh"
#include "VTS.cuh"
#include "LBMBin.cuh"
#include "writerFunction.cuh"

#endif
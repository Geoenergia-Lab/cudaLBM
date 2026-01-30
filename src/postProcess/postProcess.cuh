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
#include "../LBMTypedefs.cuh"
#include "../fileSystem.cuh"

namespace LBM
{
    namespace postProcess
    {
        __host__ [[nodiscard]] inline consteval const char *directoryPrefix() { return "postProcess"; }

        /**
         * @brief Calculates physical coordinates of lattice points
         * @tparam T Coordinate data type (typically scalar_t or double)
         * @param[in] mesh Lattice mesh providing dimensions and physical size
         * @return Vector of coordinates in interleaved format [x0, y0, z0, x1, y1, z1, ...]
         *
         * This function converts lattice indices to physical coordinates using
         * the domain dimensions stored in the mesh. Coordinates are normalized
         * to the physical domain size and distributed evenly across the lattice.
         **/
        template <typename T>
        __host__ [[nodiscard]] const std::vector<T> meshCoordinates(const host::latticeMesh &mesh)
        {
            std::vector<T> coords(mesh.nx<std::size_t>() * mesh.ny<std::size_t>() * mesh.nz<std::size_t>() * 3);

#ifdef MULTI_GPU

            static_assert(false, "postProcess::meshCoordinates not implemented for multi GPU yet");

#else
            global_for<pointLabel_t{0, 0, 0}>(
                mesh.nx<std::size_t>(), mesh.ny<std::size_t>(), mesh.nz<std::size_t>(),
                [&](const std::size_t x, const std::size_t y, const std::size_t z)
                {
                    const std::size_t idx = host::idxScalarGlobal<std::size_t>(x, y, z, mesh.nx<std::size_t>(), mesh.ny<std::size_t>());
                    // Do the conversion in double, then cast to the desired type
                    coords[3 * idx + 0] = static_cast<T>((static_cast<double>(mesh.L().x) * static_cast<double>(x * static_cast<std::size_t>(mesh.nx() > 1))) / static_cast<double>(mesh.nx<std::size_t>() - static_cast<std::size_t>(mesh.nx() > 1)));
                    coords[3 * idx + 1] = static_cast<T>((static_cast<double>(mesh.L().y) * static_cast<double>(y * static_cast<std::size_t>(mesh.ny() > 1))) / static_cast<double>(mesh.ny<std::size_t>() - static_cast<std::size_t>(mesh.ny() > 1)));
                    coords[3 * idx + 2] = static_cast<T>((static_cast<double>(mesh.L().z) * static_cast<double>(z * static_cast<std::size_t>(mesh.nz() > 1))) / static_cast<double>(mesh.nz<std::size_t>() - static_cast<std::size_t>(mesh.nz() > 1)));
                });

#endif

            return coords;
        }

        /**
         * @brief Calculates the connectivity of the points of a latticeMesh object
         * @tparam one_based If true, indices are 1-based. If false, 0-based.
         * @tparam IndexType The integer type for the connectivity data (e.g., uint32_t, uint64_t).
         * @param mesh The mesh
         * @return An std::vector of type IndexType containing the latticeMesh object connectivity
         **/
        template <const bool one_based, typename IndexType>
        __host__ [[nodiscard]] const std::vector<IndexType> meshConnectivity(const host::latticeMesh &mesh)
        {
            std::vector<IndexType> connectivity((mesh.nx<std::size_t>() - 1) * (mesh.ny<std::size_t>() - 1) * (mesh.nz<std::size_t>() - 1) * 8);
            constexpr const label_t offset = one_based ? 1 : 0;
            global_for<pointLabel_t{1, 1, 1}>(
                mesh.nx<std::size_t>(), mesh.ny<std::size_t>(), mesh.nz<std::size_t>(),
                [&](const std::size_t x, const std::size_t y, const std::size_t z)
                {
                    const std::size_t base = host::idxScalarGlobal(x, y, z, mesh.nx<std::size_t>(), mesh.ny<std::size_t>());
                    const std::size_t cell_idx = host::idxScalarGlobal(x, y, z, mesh.nx<std::size_t>() - 1, mesh.ny<std::size_t>() - 1);
                    const std::size_t stride_y = mesh.nx<std::size_t>();
                    const std::size_t stride_z = mesh.nx<std::size_t>() * mesh.ny<std::size_t>();

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
         * @tparam IndexType The integer type for the offset data (e.g., uint32_t, uint64_t).
         * @param mesh The mesh
         * @return An std::vector of type IndexType containing the latticeMesh object point offsets
         **/
        template <typename IndexType>
        __host__ [[nodiscard]] const std::vector<IndexType> meshOffsets(const host::latticeMesh &mesh)
        {
            const std::size_t nx = mesh.nx<std::size_t>();
            const std::size_t ny = mesh.ny<std::size_t>();
            const std::size_t nz = mesh.nz<std::size_t>();
            const std::size_t numElements = (nx - 1) * (ny - 1) * (nz - 1);

            std::vector<IndexType> offsets(numElements);

            for (std::size_t i = 0; i < numElements; ++i)
            {
                offsets[i] = static_cast<IndexType>((i + 1) * 8);
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

        template <typename T>
        __host__ void writeBinaryBlock(const std::vector<T> vec, std::ofstream &outFile)
        {
            const uint64_t blockSize = vec.size() * sizeof(T);

            outFile.write(reinterpret_cast<const char *>(&blockSize), sizeof(uint64_t));

            outFile.write(reinterpret_cast<const char *>(vec.data()), static_cast<std::streamsize>(blockSize));
        };
    }
}

#include "Tecplot.cuh"
#include "VTU.cuh"
#include "VTS.cuh"
#include "writerFunction.cuh"

#endif
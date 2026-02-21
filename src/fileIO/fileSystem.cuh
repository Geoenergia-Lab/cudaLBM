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
    Functions employed throughout the source code to interact with the
    file system

Namespace
    LBM

SourceFiles
    fileSystem.cuh

\*---------------------------------------------------------------------------*/

#include "../LBMIncludes.cuh"
#include "../typedefs/typedefs.cuh"

#ifndef __MBLBM_FILESYSTEM_CUH
#define __MBLBM_FILESYSTEM_CUH

namespace LBM
{
    namespace fileSystem
    {
        /**
         * @brief Enumeration for file formats
         **/
        typedef enum Enum : int
        {
            ASCII = 0,
            BINARY = 1,
            UNDEFINED = 2
        } format;

        namespace assertions
        {
            template <const format Format>
            __device__ __host__ inline consteval void validate() noexcept
            {
                static_assert((Format == ASCII) || (Format == BINARY), "Invalid format: must be ASCII or BINARY");
            }
        }

        /**
         * @brief Convert bytes to mebibytes (binary megabytes, 1024*1024 bytes)
         * @tparam T The return type for the converted value
         * @param[in] nBytes Number of bytes to convert
         * @return The equivalent value in mebibytes as type T
         **/
        template <typename T>
        __host__ [[nodiscard]] inline constexpr T to_mebibytes(const uintmax_t nBytes) noexcept
        {
            return static_cast<T>(static_cast<double>(nBytes) / (static_cast<double>(1024 * 1024)));
        }

        /**
         * @brief Gets the name of the disk on which path is found
         * @param[in] dir The directory to query
         * @return The name of the disk, e.g. "/mnt/c"
         **/
        __host__ [[nodiscard]] const name_t diskName(const std::filesystem::path &dir = std::filesystem::current_path()) noexcept
        {
            std::filesystem::path current = std::filesystem::absolute(dir);

            // Traverse up the directory tree to find a mounted Windows drive
            while (current != current.root_path())
            {
                if (current.parent_path() == current)
                {
                    break;
                }
                current = current.parent_path();
            }

            // Check if we're in a mounted Windows drive (/mnt/X/)
            if (current.string().find("/mnt/") == 0)
            {
                return current.string(); // This is already the mounted path like "/mnt/c"
            }

            // If we're in WSL filesystem, default to C: drive
            return "/mnt/c";
        }

        /**
         * @brief Gets the available storage space on the disk on which dir is located
         * @param[in] dir The directory to query
         * @return The available storage space in bytes on a dir
         **/
        __host__ [[nodiscard]] uintmax_t availableDiskSpace(const std::filesystem::path &dir = std::filesystem::current_path()) noexcept
        {
            std::error_code ec;

            const std::filesystem::space_info si = std::filesystem::space(diskName(dir), ec);

            return si.available;
        }

        /**
         * @brief Checks if there exists enough space to write nBytes in a given directory
         * @param[in] nBytes The number of bytes to be written
         * @param[in] dir The directory in which we wish to write
         * @return True if there is enough space, false otherwise
         **/
        __host__ [[nodiscard]] bool hasEnoughSpace(const uintmax_t nBytes, const std::filesystem::path &dir = std::filesystem::current_path()) noexcept
        {
            return (nBytes < availableDiskSpace(dir));
        }

        /**
         * @brief Calculate the disk space required for field data storage
         * @tparam hasFields Whether field data is present
         * @tparam Format The file format (ASCII or BINARY)
         * @param[in] nx Number of grid points in x-direction
         * @param[in] ny Number of grid points in y-direction
         * @param[in] nz Number of grid points in z-direction
         * @param[in] nVars Number of field variables
         * @return The estimated disk space required for field data in bytes
         **/
        template <const bool hasFields, const format Format>
        __host__ [[nodiscard]] inline constexpr uintmax_t fieldsDiskUsage(const uintmax_t nx, const uintmax_t ny, const uintmax_t nz, const uintmax_t nVars) noexcept
        {
            if constexpr (hasFields)
            {
                assertions::validate<Format>();

                if constexpr (Format == BINARY)
                {
                    return static_cast<uintmax_t>(nVars) * static_cast<uintmax_t>(nx) * static_cast<uintmax_t>(ny) * static_cast<uintmax_t>(nz) * static_cast<uintmax_t>(sizeof(scalar_t));
                }
                else
                {
                    if constexpr (std::is_same_v<scalar_t, double>)
                    {
                        // Handle double
                        return static_cast<uintmax_t>(nVars) * static_cast<uintmax_t>(nx) * static_cast<uintmax_t>(ny) * static_cast<uintmax_t>(nz) * 25;
                    }
                    else if constexpr (std::is_same_v<scalar_t, float>)
                    {
                        // Handle float
                        return static_cast<uintmax_t>(nVars) * static_cast<uintmax_t>(nx) * static_cast<uintmax_t>(ny) * static_cast<uintmax_t>(nz) * 15;
                    }
                }
            }
            else
            {
                return 0;
            }
        }

        /**
         * @brief Calculate the disk space required for element data storage
         * @tparam hasElements Whether element data is present
         * @tparam labelsPerElement Number of labels per element (typically 8 for hexahedra)
         * @param[in] nx Number of grid points in x-direction
         * @param[in] ny Number of grid points in y-direction
         * @param[in] nz Number of grid points in z-direction
         * @return The estimated disk space required for element data in bytes
         **/
        template <const bool hasElements, const uintmax_t labelsPerElement>
        __host__ [[nodiscard]] inline constexpr uintmax_t elementsDiskUsage(const uintmax_t nx, const uintmax_t ny, const uintmax_t nz) noexcept
        {
            if constexpr (hasElements)
            {
                return labelsPerElement * static_cast<uintmax_t>(nx - 1) * static_cast<uintmax_t>(ny - 1) * static_cast<uintmax_t>(nz - 1) * static_cast<uintmax_t>(sizeof(label_t));
            }
            else
            {
                return 0;
            }
        }

        /**
         * @brief Calculate total expected disk usage for all components
         * @tparam Format The file format (ASCII or BINARY)
         * @tparam hasFields Whether field data is present
         * @tparam hasPoints Whether point data is present
         * @tparam hasElements Whether element data is present
         * @tparam hasOffsets Whether offset data is present
         * @param[in] nx Number of grid points in x-direction
         * @param[in] ny Number of grid points in y-direction
         * @param[in] nz Number of grid points in z-direction
         * @param[in] nVars Number of field variables
         * @return The total estimated disk space required in bytes
         **/
        template <const format Format, const bool hasFields, const bool hasPoints, const bool hasElements, const bool hasOffsets>
        __host__ [[nodiscard]] inline constexpr uintmax_t expectedDiskUsage(const uintmax_t nx, const uintmax_t ny, const uintmax_t nz, const uintmax_t nVars) noexcept
        {
            return fieldsDiskUsage<hasFields, Format>(nx, ny, nz, nVars) + fieldsDiskUsage<hasPoints, Format>(nx, ny, nz, 3) + elementsDiskUsage<hasElements, 8>(nx, ny, nz) + elementsDiskUsage<hasOffsets, 1>(nx, ny, nz);
        }

        /**
         * @brief Calculate total expected disk usage for all components using mesh dimensions
         * @tparam Format The file format (ASCII or BINARY)
         * @tparam hasFields Whether field data is present
         * @tparam hasPoints Whether point data is present
         * @tparam hasElements Whether element data is present
         * @tparam hasOffsets Whether offset data is present
         * @tparam Mesh The mesh type
         * @param[in] mesh The lattice mesh
         * @param[in] nVars Number of field variables
         * @return The total estimated disk space required in bytes
         **/
        template <const format Format, const bool hasFields, const bool hasPoints, const bool hasElements, const bool hasOffsets, class LatticeMesh, typename T>
        __host__ [[nodiscard]] inline constexpr uintmax_t expectedDiskUsage(const LatticeMesh &mesh, const T nVars) noexcept
        {
            return expectedDiskUsage<Format, hasFields, hasPoints, hasElements, hasOffsets>(static_cast<uintmax_t>(mesh.template dimension<axis::X>()), static_cast<uintmax_t>(mesh.template dimension<axis::Y>()), static_cast<uintmax_t>(mesh.template dimension<axis::Z>()), static_cast<uintmax_t>(nVars));
        }

        /**
         * @brief Check if sufficient disk space is available for writing
         * @tparam Format The file format (ASCII or BINARY)
         * @tparam hasFields Whether field data is present
         * @tparam hasPoints Whether point data is present
         * @tparam hasElements Whether element data is present
         * @tparam hasOffsets Whether offset data is present
         * @tparam Mesh The mesh type
         * @param[in] mesh The lattice mesh
         * @param[in] nVars Number of field variables
         * @return True if sufficient disk space is available, false otherwise
         **/
        template <const format Format, const bool hasFields, const bool hasPoints, const bool hasElements, const bool hasOffsets, class LatticeMesh>
        __host__ [[nodiscard]] bool diskSpaceCheck(const LatticeMesh &mesh, const uintmax_t nVars)
        {
            // Calculated the approximate required space
            const uintmax_t requiredDiskSpace = expectedDiskUsage<Format, hasFields, hasPoints, hasElements, hasOffsets>(static_cast<uintmax_t>(mesh.template dimension<axis::X>()), static_cast<uintmax_t>(mesh.template dimension<axis::Y>()), static_cast<uintmax_t>(mesh.template dimension<axis::Z>()), nVars);

            // Check enough space is available
            return fileSystem::hasEnoughSpace(requiredDiskSpace);
        }

        /**
         * @brief Assert that sufficient disk space is available, throw error if not
         * @tparam Format The file format (ASCII or BINARY)
         * @tparam hasFields Whether field data is present
         * @tparam hasPoints Whether point data is present
         * @tparam hasElements Whether element data is present
         * @tparam hasOffsets Whether offset data is present
         * @tparam Mesh The mesh type
         * @param[in] mesh The lattice mesh
         * @param[in] nVars Number of field variables
         * @param[in] fileName Name of the file being written (for error message)
         * @throws std::runtime_error if insufficient disk space is available
         **/
        template <const format Format, const bool hasFields, const bool hasPoints, const bool hasElements, const bool hasOffsets, class LatticeMesh>
        __host__ void diskSpaceAssertion(const LatticeMesh &mesh, const uintmax_t nVars, const name_t &fileName)
        {
            const uintmax_t requiredDiskSpace = expectedDiskUsage<Format, hasFields, hasPoints, hasElements, hasOffsets>(static_cast<uintmax_t>(mesh.template dimension<axis::X>()), static_cast<uintmax_t>(mesh.template dimension<axis::Y>()), static_cast<uintmax_t>(mesh.template dimension<axis::Z>()), static_cast<uintmax_t>(nVars));

            if (!diskSpaceCheck<Format, hasFields, hasPoints, hasElements, hasOffsets>(mesh, static_cast<uintmax_t>(nVars)))
            {
                const uintmax_t availableSpace = fileSystem::availableDiskSpace();
                throw std::runtime_error("Insufficient disk space to write file " + fileName + "\nRequired: " + std::to_string(requiredDiskSpace) + "\nAvailable: " + std::to_string(availableSpace));
            }
        }
    }
}

#endif
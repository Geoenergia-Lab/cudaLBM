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
         * @brief File format enumeration
         **/
        typedef enum Enum : int
        {
            ASCII = 0,
            BINARY = 1,
            UNDEFINED = 2
        } format;

        namespace scalar
        {
            /**
             * @brief Disk space for scalar data
             * @tparam Fields Whether fields are included
             * @tparam Format File format (ASCII or BINARY)
             * @param[in] nx, ny, nz Number of mesh points
             * @param[in] nVars Number of variables
             */
            template <typename T, const T Present, const format Format>
            __host__ [[nodiscard]] inline constexpr host::label_t usage(const host::label_t nx, const host::label_t ny, const host::label_t nz, const host::label_t nVars) noexcept
            {
                static_assert(Format == ASCII || Format == BINARY, "Format must be ASCII or BINARY");

                if constexpr (static_cast<bool>(Present))
                {
                    const host::label_t nElements = nx * ny * nz;
                    if constexpr (Format == BINARY)
                    {
                        return nVars * nElements * sizeof(scalar_t);
                    }
                    else
                    {
                        // ASCII estimate
                        if constexpr (std::is_same_v<scalar_t, double>)
                        {
                            return nVars * nElements * 25;
                        }
                        else
                        {
                            return nVars * nElements * 15;
                        }
                    }
                }
                else
                {
                    return 0;
                }
            }
        }

        namespace connectivity
        {
            /**
             * @brief Disk space for element connectivity
             * @tparam Elements Whether elements are included
             * @tparam N Number of labels per element
             */
            template <typename T, const T Present, const host::label_t N>
            __host__ [[nodiscard]] inline constexpr host::label_t usage(const host::label_t nx, const host::label_t ny, const host::label_t nz) noexcept
            {
                if constexpr (static_cast<bool>(Present))
                {
                    return N * (nx - 1) * (ny - 1) * (nz - 1) * sizeof(device::label_t);
                }
                else
                {
                    return 0;
                }
            }
        }

        namespace fields
        {
            typedef enum Enum : bool
            {
                No = false,
                Yes = true
            } contained;

            /**
             * @brief Disk space for field data (multiple variables per grid point)
             * @tparam Fields Whether fields are included
             * @tparam Format File format (ASCII or BINARY)
             * @param[in] nx, ny, nz Mesh dimensions
             * @param[in] nVars Number of variables
             */
            template <const contained Fields, const format Format>
            __host__ [[nodiscard]] inline constexpr host::label_t usage(const host::label_t nx, const host::label_t ny, const host::label_t nz, const host::label_t nVars) noexcept
            {
                return scalar::usage<contained, Fields, Format>(nx, ny, nz, nVars);
            }
        }

        namespace points
        {
            typedef enum Enum : bool
            {
                No = false,
                Yes = true
            } contained;

            /**
             * @brief Disk space for point coordinates (always 3 components per node)
             * @tparam Points Whether points are included
             * @tparam Format File format
             * @param[in] nx, ny, nz Mesh dimensions
             */
            template <const contained Points, const format Format>
            __host__ [[nodiscard]] inline constexpr host::label_t usage(const host::label_t nx, const host::label_t ny, const host::label_t nz) noexcept
            {
                return scalar::usage<contained, Points, Format>(nx, ny, nz, 3);
            }
        }

        namespace elements
        {
            typedef enum Enum : bool
            {
                No = false,
                Yes = true
            } contained;

            /**
             * @brief Disk space for element connectivity (8 labels per element)
             * @tparam Elements Whether elements are included
             * @param[in] nx, ny, nz Mesh dimensions
             */
            template <const contained Elements>
            __host__ [[nodiscard]] inline constexpr host::label_t usage(const host::label_t nx, const host::label_t ny, const host::label_t nz) noexcept
            {
                return connectivity::usage<contained, Elements, 8>(nx, ny, nz);
            }
        }

        namespace offsets
        {
            typedef enum Enum : bool
            {
                No = false,
                Yes = true
            } contained;

            /**
             * @brief Disk space for offset data (1 label per element)
             * @tparam Offsets Whether offsets are included
             * @param[in] nx, ny, nz Mesh dimensions
             */
            template <const contained Offsets>
            __host__ [[nodiscard]] inline constexpr host::label_t usage(const host::label_t nx, const host::label_t ny, const host::label_t nz) noexcept
            {
                return connectivity::usage<contained, Offsets, 1>(nx, ny, nz);
            }
        }

        /**
         * @brief Compute total expected disk space for all output components
         * @tparam Format File format
         * @tparam fields Whether field data is included
         * @tparam points Whether point coordinates are included
         * @tparam elements Whether element connectivity is included
         * @tparam offsets Whether offset data is included
         * @param[in] nx, ny, nz Mesh dimensions
         * @param[in] nVars Number of field variables (ignored if fields == No)
         * @return Total bytes required
         */
        template <const format Format, const fields::contained Fields, const points::contained Points, const elements::contained Elements, const offsets::contained Offsets>
        __host__ [[nodiscard]] inline constexpr host::label_t expectedDiskUsage(const host::label_t nx, const host::label_t ny, const host::label_t nz, const host::label_t nVars) noexcept
        {
            return fields::usage<Fields, Format>(nx, ny, nz, nVars) + points::usage<Points, Format>(nx, ny, nz) + elements::usage<Elements>(nx, ny, nz) + offsets::usage<Offsets>(nx, ny, nz);
        }

        /**
         * @brief Overload for usage with a mesh object.
         * @tparam Format,fields, points, elements, offsets Same as above.
         * @tparam LatticeMesh Type of the mesh (must provide dimension<axis::X, host::label_t>(), etc.).
         * @param[in] mesh The lattice mesh.
         * @param[in] nVars Number of field variables.
         * @return Total bytes required.
         */
        template <const format Format, const fields::contained Fields, const points::contained Points, const elements::contained Elements, const offsets::contained Offsets, class LatticeMesh>
        __host__ [[nodiscard]] inline constexpr host::label_t expectedDiskUsage(const LatticeMesh &mesh, const host::label_t nVars) noexcept
        {
            return expectedDiskUsage<Format, Fields, Points, Elements, Offsets>(static_cast<host::label_t>(mesh.template dimension<axis::X>()), static_cast<host::label_t>(mesh.template dimension<axis::Y>()), static_cast<host::label_t>(mesh.template dimension<axis::Z>()), nVars);
        }

        /**
         * @brief Gets the available storage space on the disk containing dir.
         * @param[in] dir The directory to query
         * @return Available bytes.
         */
        __host__ [[nodiscard]] inline host::label_t availableDiskSpace(const std::filesystem::path &dir = std::filesystem::current_path()) noexcept
        {
            std::error_code ec;
            const std::filesystem::space_info space = std::filesystem::space(dir, ec);
            return ec ? 0 : space.available;
        }

        /**
         * @brief Checks if at least `required` bytes are available on the disk of `dir`.
         */
        __host__ [[nodiscard]] inline bool hasEnoughSpace(const host::label_t required, const std::filesystem::path &dir = std::filesystem::current_path()) noexcept
        {
            return required < availableDiskSpace(dir);
        }

        /**
         * @brief Throws a runtime_error due to insufficient disk space
         */
        __host__ void insufficientDiskSpace(const name_t &fileName, const host::label_t expected, const host::label_t available = availableDiskSpace())
        {
            throw std::runtime_error("Insufficient disk space to write " + fileName + "\nRequired: " + std::to_string(expected) + "\nAvailable: " + std::to_string(available));
        }

        /**
         * @brief Throws if insufficient disk space.
         */
        __host__ void ensureDiskSpace(const name_t &fileName, const host::label_t required, const host::label_t available = availableDiskSpace())
        {
            if (required > available)
            {
                insufficientDiskSpace(fileName, required, available);
            }
        }

        /**
         * @brief Check whether sufficient disk space exists for a given output config.
         */
        template <const format Format, const fields::contained Fields, const points::contained Points, const elements::contained Elements, const offsets::contained Offsets, class LatticeMesh>
        __host__ [[nodiscard]] bool diskSpaceCheck(const LatticeMesh &mesh, const host::label_t nVars, const std::filesystem::path &dir = std::filesystem::current_path()) noexcept
        {
            const host::label_t needed = expectedDiskUsage<Format, Fields, Points, Elements, Offsets>(mesh, nVars);
            return hasEnoughSpace(needed, dir);
        }

        /**
         * @brief Assert (throw) that sufficient disk space exists.
         */
        template <const format Format, const fields::contained Fields, const points::contained Points, const elements::contained Elements, const offsets::contained Offsets, class LatticeMesh>
        __host__ void diskSpaceAssertion(const LatticeMesh &mesh, const host::label_t nVars, const name_t &fileName, const std::filesystem::path &dir = std::filesystem::current_path())
        {
            const host::label_t needed = expectedDiskUsage<Format, Fields, Points, Elements, Offsets>(mesh, nVars);

            ensureDiskSpace(fileName, needed, availableDiskSpace(dir));
        }

        /**
         * @brief Convert bytes to mebibytes
         * @tparam T Return type
         * @param[in] bytes Number of bytes
         */
        template <typename T>
        __host__ [[nodiscard]] inline constexpr T to_mebibytes(const host::label_t bytes) noexcept
        {
            return static_cast<T>(static_cast<double>(bytes) / (1024.0 * 1024.0));
        }

    }
}

#endif
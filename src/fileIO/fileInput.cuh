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
    Implementation of reading solution variables encoded in binary format

Namespace
    LBM::fileIO

SourceFiles
    fileInput.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_FILEINPUT_CUH
#define __MBLBM_FILEINPUT_CUH

namespace LBM
{
    namespace fileIO
    {
        /**
         * @brief Swap endianness for a single value
         * @tparam T Data type of value to swap
         * @param[in,out] value Reference to value whose endianness will be swapped
         **/
        template <typename T>
        void swapEndian(T &value)
        {
            char *bytes = reinterpret_cast<char *>(&value);
            for (std::size_t i = 0; i < sizeof(T) / 2; ++i)
            {
                std::swap(bytes[i], bytes[sizeof(T) - 1 - i]);
            }
        }

        /**
         * @brief Swap endianness for all values in a vector
         * @tparam T Data type of vector elements
         * @param[in,out] data Vector whose elements' endianness will be swapped
         **/
        template <typename T>
        void swapEndianVector(std::vector<T> &data)
        {
            for (T &value : data)
            {
                swapEndian(value);
            }
        }

        /**
         * @brief Read all field data from a binary file
         * @tparam T Floating-point type to read (must match file format)
         * @param[in] fileName Name of the file to read
         * @return Vector containing all field data in AoS (Array of Structures) format
         * @throws std::runtime_error if file doesn't exist, is inaccessible, or has format issues
         *
         * This function reads the entire binary data from a field file, handling
         * endianness conversion if necessary. Data is returned in AoS format where
         * all variables for each point are stored contiguously.
         **/
        template <typename T>
        __host__ [[nodiscard]] const std::vector<T> readFieldFile(const name_t &fileName)
        {
            endian::assertions::validate();
            types::assertions::validate<T>();

            // Check that the file exists - if it doesn't, throw an exception
            if (!std::filesystem::exists(fileName))
            {
                throw std::runtime_error("File does not exist: " + fileName);
            }

            // Parse header metadata
            const fieldFileHeader header = parseFieldFileHeader(fileName);

            // Validate scalar size
            if (sizeof(T) != header.scalarSize)
            {
                throw std::runtime_error("Scalar size mismatch between file and template type");
            }

            // Calculate expected data size
            const std::size_t totalPoints = header.nx * header.ny * header.nz;
            const std::size_t totalDataCount = totalPoints * header.nVars;

            // Open file and jump to binary data
            std::ifstream in(fileName, std::ios::binary);
            if (!in)
            {
                throw std::runtime_error("Cannot open file: " + fileName);
            }

            // Safe conversion for seekg
            if (header.dataStartPos > static_cast<std::size_t>(std::numeric_limits<std::streamoff>::max()))
            {
                throw std::runtime_error("File position overflow");
            }
            in.seekg(static_cast<std::streamoff>(header.dataStartPos));

            // Read binary data
            std::vector<T> data(totalDataCount);
            const std::size_t byteCount = totalDataCount * sizeof(T);

            // Check for streamsize overflow
            if (byteCount > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()))
            {
                throw std::runtime_error("Data size exceeds maximum stream size");
            }
            in.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(byteCount));

            if (!in.good() || in.gcount() != static_cast<std::streamsize>(byteCount))
            {
                throw std::runtime_error("Error reading binary data");
            }

            // Handle endianness conversion if needed
            const bool systemIsLittle = (std::endian::native == std::endian::little);
            if (systemIsLittle != header.isLittleEndian)
            {
                swapEndianVector(data);
            }

            return data;
        }

        /**
         * @brief Read specific field data from a binary file
         * @tparam T Floating-point type to read (must match file format)
         * @param[in] fileName Name of the file to read
         * @param[in] fieldName Name of the specific field to extract
         * @return Vector containing data for the requested field in SoA (Structure of Arrays) format
         * @throws std::runtime_error if field doesn't exist or file has format issues
         *
         * This function extracts a single field from a multi-field binary file,
         * handling endianness conversion and validation. It's more efficient than
         * reading the entire file when only specific fields are needed.
         **/
        template <typename T>
        __host__ [[nodiscard]] const std::vector<T> readFieldByName(const name_t &fileName, const name_t &fieldName)
        {
            types::assertions::validate<T>();
            endian::assertions::validate();

            // Check if file exists
            if (!std::filesystem::exists(fileName))
            {
                throw std::runtime_error("File does not exist: " + fileName);
            }

            // Get file size for validation
            std::ifstream sizeCheck(fileName, std::ios::binary | std::ios::ate);
            if (!sizeCheck)
            {
                throw std::runtime_error("Cannot open file for size check: " + fileName);
            }
            const std::size_t fileSize = static_cast<std::size_t>(sizeCheck.tellg());
            sizeCheck.close();

            // Parse header to get file structure and field names
            const fieldFileHeader header = parseFieldFileHeader(fileName);

            // Validate scalar size
            if (sizeof(T) != header.scalarSize)
            {
                throw std::runtime_error("Scalar size mismatch between file (" + std::to_string(header.scalarSize) + " bytes) and template type (" + std::to_string(sizeof(T)) + " bytes)");
            }

            // Find the requested field name
            const auto it = std::find(header.fieldNames.begin(), header.fieldNames.end(), fieldName);
            if (it == header.fieldNames.end())
            {
                // Create a list of available field names for better error message
                name_t availableFields;
                for (const auto &name : header.fieldNames)
                {
                    if (!availableFields.empty())
                    {
                        availableFields += ", ";
                    }
                    availableFields += "'" + name + "'";
                }
                throw std::runtime_error("Field name '" + fieldName + "' not found in file. Available fields: " + availableFields);
            }

            // Calculate field index and data position - FIXED sign conversion issue
            const std::ptrdiff_t signedFieldIndex = std::distance(header.fieldNames.begin(), it);
            if (signedFieldIndex < 0)
            {
                throw std::runtime_error("Internal error: Negative field index");
            }

            const std::size_t fieldIndex = static_cast<std::size_t>(signedFieldIndex);
            const std::size_t pointsPerField = header.nx * header.ny * header.nz;

            // Check for potential overflow in calculations
            if (pointsPerField == 0)
            {
                throw std::runtime_error("Invalid field dimensions: points per field is zero");
            }

            // Check if fieldIndex is valid
            if (fieldIndex >= header.nVars)
            {
                throw std::runtime_error("Field index out of range");
            }

            // Check for overflow in fieldOffset calculation
            if (fieldIndex > (std::numeric_limits<std::size_t>::max() / pointsPerField / sizeof(T)))
            {
                throw std::runtime_error("Field offset calculation would overflow");
            }

            const std::size_t fieldOffset = fieldIndex * pointsPerField * sizeof(T);

            // Check if fieldOffset would exceed file bounds
            if (fieldOffset > (std::numeric_limits<std::size_t>::max() - header.dataStartPos))
            {
                throw std::runtime_error("Field start position calculation would overflow");
            }

            const std::size_t fieldStartPos = header.dataStartPos + fieldOffset;

            // Check if field data would extend beyond file end
            if (fieldStartPos > fileSize)
            {
                throw std::runtime_error("Field start position is beyond file end");
            }

            const std::size_t fieldByteSize = pointsPerField * sizeof(T);

            // Check if field data would extend beyond file end
            if (fieldStartPos > (std::numeric_limits<std::size_t>::max() - fieldByteSize))
            {
                throw std::runtime_error("Field end position calculation would overflow");
            }

            if (fieldStartPos + fieldByteSize > fileSize)
            {
                throw std::runtime_error("Field data extends beyond file end");
            }

            // Open file and jump to field data
            std::ifstream in(fileName, std::ios::binary);
            if (!in)
            {
                throw std::runtime_error("Cannot open file: " + fileName);
            }

            // Check for position overflow
            if (fieldStartPos > static_cast<std::size_t>(std::numeric_limits<std::streamoff>::max()))
            {
                throw std::runtime_error("Field position overflow");
            }

            in.seekg(static_cast<std::streamoff>(fieldStartPos));
            if (!in.good())
            {
                throw std::runtime_error("Failed to seek to field position");
            }

            // Read field data
            std::vector<T> fieldData(pointsPerField);
            const std::size_t byteCount = fieldByteSize;

            // Check for streamsize overflow
            if (byteCount > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()))
            {
                throw std::runtime_error("Field data size exceeds maximum stream size");
            }

            in.read(reinterpret_cast<char *>(fieldData.data()), static_cast<std::streamsize>(byteCount));

            if (!in.good())
            {
                throw std::runtime_error("Error reading field data: stream not good after read");
            }

            if (in.gcount() != static_cast<std::streamsize>(byteCount))
            {
                throw std::runtime_error("Incomplete field data read. Expected " + std::to_string(byteCount) + " bytes, got " + std::to_string(in.gcount()) + " bytes");
            }

            // Handle endianness conversion if needed
            const bool systemIsLittle = (std::endian::native == std::endian::little);
            if (systemIsLittle != header.isLittleEndian)
            {
                swapEndianVector(fieldData);
            }

            return fieldData;
        }

        /**
         * @brief Convert Array of Structures (AoS) to Structure of Arrays (SoA)
         * @tparam T Data type of array elements
         * @tparam M Mesh type providing dimension information
         * @param[in] fMom Input data in AoS format (all variables interleaved per point)
         * @param[in] mesh The lattice mesh
         * @return Vector of vectors where each inner vector contains all values for one variable
         * @throws std::invalid_argument if input size doesn't match mesh dimensions
         *
         * This function reorganizes data from AoS format (where all variables for
         * each point are stored together) to SoA format (where each variable's values
         * are stored in separate contiguous arrays).
         **/
        template <typename T, class LatticeMesh>
        __host__ [[nodiscard]] const std::vector<std::vector<T>> deinterleaveAoS(const std::vector<T> &fMom, const LatticeMesh &mesh)
        {
            const std::size_t nNodes = mesh.template dimension<axis::X, std::size_t>() * mesh.template dimension<axis::Y, std::size_t>() * mesh.template dimension<axis::Z, std::size_t>();
            if (fMom.size() % nNodes != 0)
            {
                throw std::invalid_argument("fMom size (" + std::to_string(fMom.size()) + ") is not divisible by mesh points (" + std::to_string(nNodes) + ")");
            }
            const std::size_t nFields = fMom.size() / nNodes;

            std::vector<std::vector<T>> soa(nFields, std::vector<T>(nNodes, 0));

            const std::size_t nxGPUs = mesh.template nDevices<axis::X, std::size_t>();
            const std::size_t nyGPUs = mesh.template nDevices<axis::Y, std::size_t>();
            const std::size_t nzGPUs = mesh.template nDevices<axis::Z, std::size_t>();

            const std::size_t nxBlocksPerDevice = mesh.template nBlocks<axis::X, std::size_t>() / nxGPUs;
            const std::size_t nyBlocksPerDevice = mesh.template nBlocks<axis::Y, std::size_t>() / nyGPUs;
            const std::size_t nzBlocksPerDevice = mesh.template nBlocks<axis::Z, std::size_t>() / nzGPUs;

            const std::size_t pointsPerBlock = block::size<std::size_t>();
            const std::size_t nPointsPerDevice = nxBlocksPerDevice * nyBlocksPerDevice * nzBlocksPerDevice * pointsPerBlock;

            GPU::forAll(
                mesh.nDevices(),
                [&](const std::size_t GPU_x, const std::size_t GPU_y, const std::size_t GPU_z)
                {
                    const std::size_t virtualDeviceIndex = GPU::idx<std::size_t>(GPU_x, GPU_y, GPU_z, nxGPUs, nyGPUs);

                    host::forAll(
                        mesh.blocksPerDevice(),
                        [&](const std::size_t bx, const std::size_t by, const std::size_t bz,
                            const std::size_t tx, const std::size_t ty, const std::size_t tz)
                        {
                            // Global coordinates (for output)
                            const std::size_t x = (GPU_x * nxBlocksPerDevice + bx) * block::nx<std::size_t>() + tx;
                            const std::size_t y = (GPU_y * nyBlocksPerDevice + by) * block::ny<std::size_t>() + ty;
                            const std::size_t z = (GPU_z * nzBlocksPerDevice + bz) * block::nz<std::size_t>() + tz;

                            const std::size_t idxGlobal = global::idx(x, y, z, mesh.template dimension<axis::X, std::size_t>(), mesh.template dimension<axis::Y, std::size_t>());

                            // Local index within this GPU's storage (block‑major order)
                            const std::size_t blockLin = (bz * nyBlocksPerDevice + by) * nxBlocksPerDevice + bx;
                            const std::size_t threadLin = (tz * block::ny<std::size_t>() + ty) * block::nx<std::size_t>() + tx;
                            const std::size_t localIdx = blockLin * pointsPerBlock + threadLin;

                            for (std::size_t field = 0; field < nFields; field++)
                            {
                                const std::size_t srcIdx = field * nNodes + virtualDeviceIndex * nPointsPerDevice + localIdx;
                                soa[field][idxGlobal] = fMom[srcIdx];
                            }
                        });
                });

            return soa;
        }

    }

}

#endif
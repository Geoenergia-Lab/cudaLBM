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
         * @struct fieldFileHeader
         * @brief Contains metadata extracted from field file headers
         *
         * This structure holds all the metadata required to interpret
         * binary field data files, including dimensions, data format,
         * and field information.
         **/
        struct fieldFileHeader
        {
            const bool isLittleEndian;      //!< Endianness of the binary data
            const std::size_t scalarSize;   //!< Size of scalar values (4 or 8 bytes)
            const std::size_t nx;           //!< Grid dimension in x-direction
            const std::size_t ny;           //!< Grid dimension in y-direction
            const std::size_t nz;           //!< Grid dimension in z-direction
            const std::size_t nVars;        //!< Number of variables per grid point
            const std::size_t dataStartPos; //!< File position where binary data begins
            const words_t fieldNames;       //!< Names of all field variables
        };

        /**
         * @brief Trims leading/trailing whitespace and trailing semicolons from a string
         * @param[in] str The input string to trim
         * @return The trimmed string
         **/
        __host__ [[nodiscard]] const name_t filestring_trim(const name_t &str)
        {
            const std::size_t start = str.find_first_not_of(" \t\r\n");
            const std::size_t end = str.find_last_not_of(" \t\r\n;");
            return (start == name_t::npos) ? "" : str.substr(start, end - start + 1);
        }

        /**
         * @brief Parse header metadata from field file
         * @param[in] fileName Name of the file to parse
         * @return Parsed header information
         * @throws std::runtime_error if file doesn't exist, is inaccessible, or has invalid format
         *
         * This function reads and validates the header section of field files,
         * extracting metadata about grid dimensions, data format, and field names.
         * It performs comprehensive error checking for file integrity and format compliance.
         **/
        __host__ [[nodiscard]] const fieldFileHeader parseFieldFileHeader(const name_t &fileName)
        {
            // Check if file exists and is accessible
            if (!std::filesystem::exists(fileName))
            {
                throw std::runtime_error("File does not exist: " + fileName);
            }

            std::ifstream in(fileName, std::ios::binary);
            if (!in)
            {
                throw std::runtime_error("Cannot open file: " + fileName);
            }

            // Get file size for validation
            in.seekg(0, std::ios::end);
            const auto fileSizePos = in.tellg();
            if (fileSizePos == -1)
            {
                throw std::runtime_error("Cannot determine file size");
            }

            // Safe cast: file size should be non-negative
            if (fileSizePos < 0)
            {
                throw std::runtime_error("Invalid file size (negative)");
            }

            const std::size_t fileSize = static_cast<std::size_t>(fileSizePos);
            in.seekg(0, std::ios::beg);

            name_t line;
            bool inSystemInfo = false;
            bool inFieldData = false;
            bool inFieldInfo = false;
            bool inFieldNames = false;
            bool isLittleEndian = false;

            // Variables to store parsed data
            std::size_t scalarSize = 0;
            std::size_t nx = 0;
            std::size_t ny = 0;
            std::size_t nz = 0;
            std::size_t nVars = 0;
            std::size_t totalPoints = 0;
            std::size_t dataStartPos = 0;
            words_t fieldNamesVec;
            std::size_t expectedFieldCount = 0;
            bool foundFieldData = false;
            bool foundSystemInfo = false;
            bool foundFieldInfo = false;

            // Track which sections we've already seen to detect duplicates
            bool systemInfoSeen = false;
            bool fieldDataSeen = false;
            bool fieldInfoSeen = false;

            // Track line number for better error messages
            std::size_t lineNumber = 0;

            while (std::getline(in, line))
            {
                lineNumber++;
                line = filestring_trim(line);
                if (line.empty())
                {
                    continue;
                }

                // Detect sections (regardless of order)
                if (line == "systemInformation")
                {
                    if (systemInfoSeen)
                    {
                        throw std::runtime_error("Duplicate systemInformation section at line " + std::to_string(lineNumber));
                    }
                    inSystemInfo = true;
                    foundSystemInfo = true;
                    systemInfoSeen = true;
                    continue;
                }
                else if (line == "fieldData")
                {
                    if (fieldDataSeen)
                    {
                        throw std::runtime_error("Duplicate fieldData section at line " + std::to_string(lineNumber));
                    }
                    inFieldData = true;
                    fieldDataSeen = true;
                    continue;
                }
                else if (line == "fieldInformation")
                {
                    if (fieldInfoSeen)
                    {
                        throw std::runtime_error("Duplicate fieldInformation section at line " + std::to_string(lineNumber));
                    }
                    inFieldInfo = true;
                    foundFieldInfo = true;
                    fieldInfoSeen = true;
                    continue;
                }

                // Parse systemInformation section
                if (inSystemInfo)
                {
                    if (line == "}")
                    {
                        inSystemInfo = false;
                    }
                    else if (line.find("binaryType") != name_t::npos)
                    {
                        isLittleEndian = (line.find("littleEndian") != name_t::npos);
                    }
                    else if (line.find("scalarType") != name_t::npos)
                    {
                        if (line.find("32 bit") != name_t::npos)
                        {
                            scalarSize = 4;
                        }
                        else if (line.find("64 bit") != name_t::npos)
                        {
                            scalarSize = 8;
                        }
                        else
                        {
                            throw std::runtime_error("Invalid scalarType at line " + std::to_string(lineNumber));
                        }
                    }
                }

                // Parse fieldData section for dimensions
                if (inFieldData && !foundFieldData)
                {
                    if (line == "}")
                    {
                        inFieldData = false;
                    }
                    else if (line.find("field[") != name_t::npos)
                    {
                        // Extract dimensions from pattern: field[total][nx][ny][nz][nVars]
                        std::vector<std::size_t> dims;
                        std::size_t pos = 0;

                        while ((pos = line.find('[', pos)) != name_t::npos)
                        {
                            const std::size_t end = line.find(']', pos);
                            if (end == name_t::npos)
                            {
                                throw std::runtime_error("Unclosed bracket at line " + std::to_string(lineNumber));
                            }

                            try
                            {
                                const name_t dimStr = line.substr(pos + 1, end - pos - 1);
                                const unsigned long long dimValue = std::stoull(dimStr);

                                // Check for overflow before casting to std::size_t
                                if (dimValue > std::numeric_limits<std::size_t>::max())
                                {
                                    throw std::runtime_error("Dimension value too large at line " + std::to_string(lineNumber));
                                }

                                dims.push_back(static_cast<std::size_t>(dimValue));
                            }
                            catch (const std::out_of_range &)
                            {
                                throw std::runtime_error("Dimension value out of range at line " + std::to_string(lineNumber));
                            }
                            catch (...)
                            {
                                throw std::runtime_error("Invalid dimension format at line " + std::to_string(lineNumber));
                            }
                            pos = end + 1;
                        }

                        if (dims.size() < 5)
                        {
                            throw std::runtime_error("Invalid field dimensions at line " + std::to_string(lineNumber));
                        }

                        totalPoints = dims[0];
                        nVars = dims[1];
                        nz = dims[2];
                        ny = dims[3];
                        nx = dims[4];

                        // Validate dimensions
                        if (nx == 0 || ny == 0 || nz == 0 || nVars == 0)
                        {
                            throw std::runtime_error("Invalid grid dimensions: nx, ny, nz, nVars cannot be zero at line " + std::to_string(lineNumber));
                        }

                        // Check for potential overflow in multiplication
                        if (nx > std::numeric_limits<std::size_t>::max() / ny / nz / nVars)
                        {
                            throw std::runtime_error("Dimension product would overflow at line " + std::to_string(lineNumber));
                        }

                        if (totalPoints != nx * ny * nz * nVars)
                        {
                            throw std::runtime_error("Dimension mismatch at line " + std::to_string(lineNumber) + ": total points (" + std::to_string(totalPoints) + ") != nx * ny * nz * nVars (" + std::to_string(nx * ny * nz * nVars) + ")");
                        }

                        // Skip next line (contains "{")
                        if (!std::getline(in, line))
                        {
                            throw std::runtime_error("Unexpected end of file after field declaration");
                        }
                        lineNumber++;

                        // Record start position of binary data with safety check
                        const auto dataPos = in.tellg();
                        if (dataPos == -1)
                        {
                            throw std::runtime_error("Error getting file position");
                        }

                        if (dataPos < 0)
                        {
                            throw std::runtime_error("Invalid file position (negative)");
                        }

                        // Check for overflow before casting to std::size_t
                        if (static_cast<unsigned long long>(dataPos) > std::numeric_limits<std::size_t>::max())
                        {
                            throw std::runtime_error("File position too large for std::size_t");
                        }

                        dataStartPos = static_cast<std::size_t>(dataPos);

                        // Check if data start position is within file bounds
                        if (dataStartPos > fileSize)
                        {
                            throw std::runtime_error("Data start position exceeds file size");
                        }

                        foundFieldData = true;
                        inFieldData = false;
                    }
                }

                // Parse fieldInformation section for field names
                if (inFieldInfo)
                {
                    if (line == "}")
                    {
                        inFieldInfo = false;
                    }
                    else if (line.find("fieldNames[") != name_t::npos)
                    {
                        // Extract expected number of field names
                        const std::size_t startBracket = line.find('[');
                        const std::size_t endBracket = line.find(']');
                        if (startBracket == name_t::npos || endBracket == name_t::npos)
                        {
                            throw std::runtime_error("Invalid fieldNames format at line " + std::to_string(lineNumber));
                        }
                        try
                        {
                            const unsigned long long count = std::stoull(
                                line.substr(startBracket + 1, endBracket - startBracket - 1));

                            // Check for overflow before casting to std::size_t
                            if (count > std::numeric_limits<std::size_t>::max())
                            {
                                throw std::runtime_error("Field names count too large at line " + std::to_string(lineNumber));
                            }

                            expectedFieldCount = static_cast<std::size_t>(count);
                            if (expectedFieldCount == 0)
                            {
                                throw std::runtime_error("Field names count cannot be zero at line " + std::to_string(lineNumber));
                            }
                        }
                        catch (const std::out_of_range &)
                        {
                            throw std::runtime_error("Field names count out of range at line " + std::to_string(lineNumber));
                        }
                        catch (...)
                        {
                            throw std::runtime_error("Invalid fieldNames count at line " + std::to_string(lineNumber));
                        }
                        // Enter fieldNames block
                        inFieldNames = true;

                        // Skip the opening brace line
                        if (!std::getline(in, line))
                        {
                            throw std::runtime_error("Unexpected end of file after fieldNames declaration");
                        }
                        lineNumber++;
                        line = filestring_trim(line);
                        if (line != "{")
                        {
                            throw std::runtime_error("Expected opening brace after fieldNames declaration at line " + std::to_string(lineNumber));
                        }
                    }
                    else if (inFieldNames)
                    {
                        if (line == "}")
                        {
                            inFieldNames = false;
                            inFieldInfo = false;
                        }
                        else if (line != "{") // Skip the opening brace if we encounter it
                        {
                            // Remove trailing semicolon and trim to get field name
                            if (line.back() == ';')
                            {
                                line.pop_back();
                            }
                            const name_t fieldName = filestring_trim(line);

                            // Validate field name
                            if (fieldName.empty())
                            {
                                throw std::runtime_error("Empty field name at line " + std::to_string(lineNumber));
                            }

                            // Check for duplicate field names
                            // if (std::find(fieldNamesVec.begin(), fieldNamesVec.end(), fieldName) != fieldNamesVec.end())
                            if (string::containsString(fieldNamesVec, fieldName))
                            {
                                throw std::runtime_error("Duplicate field name '" + fieldName + "' at line " + std::to_string(lineNumber));
                            }

                            fieldNamesVec.push_back(fieldName);

                            // Check if we've exceeded the expected number of field names
                            if (fieldNamesVec.size() > expectedFieldCount)
                            {
                                throw std::runtime_error("Too many field names at line " + std::to_string(lineNumber) + ". Expected: " + std::to_string(expectedFieldCount));
                            }
                        }
                    }
                }

                // Early exit if we've collected all necessary information
                if (scalarSize > 0 && foundFieldData && fieldNamesVec.size() == expectedFieldCount && expectedFieldCount > 0)
                {
                    break;
                }
            }

            // Final validation checks
            if (!foundSystemInfo)
            {
                throw std::runtime_error("Missing systemInformation section");
            }

            if (!foundFieldInfo)
            {
                throw std::runtime_error("Missing fieldInformation section");
            }

            if (!foundFieldData)
            {
                throw std::runtime_error("Missing fieldData section");
            }

            if (scalarSize == 0)
            {
                throw std::runtime_error("Missing or invalid scalarType in systemInformation");
            }

            if (fieldNamesVec.size() != nVars)
            {
                throw std::runtime_error("Field names count (" + std::to_string(fieldNamesVec.size()) + ") does not match nVars (" + std::to_string(nVars) + ")");
            }

            if (fieldNamesVec.size() != expectedFieldCount)
            {
                throw std::runtime_error("Field names count (" + std::to_string(fieldNamesVec.size()) + ") does not match declared count (" + std::to_string(expectedFieldCount) + ")");
            }

            // Check if binary data size matches expectations
            // Check for potential overflow in multiplication
            if (totalPoints > std::numeric_limits<std::size_t>::max() / scalarSize)
            {
                throw std::runtime_error("Data size calculation would overflow");
            }

            const std::size_t expectedDataSize = totalPoints * scalarSize;
            if (fileSize - dataStartPos < expectedDataSize)
            {
                throw std::runtime_error("Insufficient data in file. Expected " + std::to_string(expectedDataSize) + " bytes, but only " + std::to_string(fileSize - dataStartPos) + " bytes available");
            }

            return {isLittleEndian, scalarSize, nx, ny, nz, nVars, dataStartPos, fieldNamesVec};
        }

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
            static_assert(std::is_floating_point_v<T>, "T must be floating point");
            static_assert(std::endian::native == std::endian::little || std::endian::native == std::endian::big, "System must be little or big endian");

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
            static_assert(std::is_floating_point_v<T>, "T must be floating point");
            static_assert(std::endian::native == std::endian::little || std::endian::native == std::endian::big, "System must be little or big endian");

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
         * @param[in] mesh Mesh object providing dimension information
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
            const std::size_t nNodes = mesh.template nx<std::size_t>() * mesh.template ny<std::size_t>() * mesh.template nz<std::size_t>();
            if (fMom.size() % nNodes != 0)
            {
                throw std::invalid_argument("fMom size (" + std::to_string(fMom.size()) + ") is not divisible by mesh points (" + std::to_string(nNodes) + ")");
            }
            const std::size_t nFields = fMom.size() / nNodes;

            std::vector<std::vector<T>> soa(nFields, std::vector<T>(nNodes, 0));

            const std::size_t nxGPUs = mesh.template nDevices<axis::X, std::size_t>();
            const std::size_t nyGPUs = mesh.template nDevices<axis::Y, std::size_t>();
            const std::size_t nzGPUs = mesh.template nDevices<axis::Z, std::size_t>();

            const std::size_t nxBlocksPerGPU = mesh.template nxBlocks<std::size_t>() / nxGPUs;
            const std::size_t nyBlocksPerGPU = mesh.template nyBlocks<std::size_t>() / nyGPUs;
            const std::size_t nzBlocksPerGPU = mesh.template nzBlocks<std::size_t>() / nzGPUs;

            const std::size_t pointsPerBlock = block::size<std::size_t>();
            const std::size_t nPointsPerGPU = nxBlocksPerGPU * nyBlocksPerGPU * nzBlocksPerGPU * pointsPerBlock;

            gpu_for(
                nxGPUs, nyGPUs, nzGPUs,
                [&](const std::size_t GPU_x, const std::size_t GPU_y, const std::size_t GPU_z)
                {
                    const std::size_t virtualDeviceIndex = GPU::idx<std::size_t>(GPU_x, GPU_y, GPU_z, nxGPUs, nyGPUs);

                    grid_for(
                        nxBlocksPerGPU, nyBlocksPerGPU, nzBlocksPerGPU,
                        [&](const std::size_t bx, const std::size_t by, const std::size_t bz,
                            const std::size_t tx, const std::size_t ty, const std::size_t tz)
                        {
                            // Global coordinates (for output)
                            const std::size_t x = (GPU_x * nxBlocksPerGPU + bx) * block::nx<std::size_t>() + tx;
                            const std::size_t y = (GPU_y * nyBlocksPerGPU + by) * block::ny<std::size_t>() + ty;
                            const std::size_t z = (GPU_z * nzBlocksPerGPU + bz) * block::nz<std::size_t>() + tz;

                            const std::size_t idxGlobal = global::idx(x, y, z, mesh.template nx<std::size_t>(), mesh.template ny<std::size_t>());

                            // Local index within this GPU's storage (block‑major order)
                            const std::size_t blockLin = (bz * nyBlocksPerGPU + by) * nxBlocksPerGPU + bx;
                            const std::size_t threadLin = (tz * block::ny<std::size_t>() + ty) * block::nx<std::size_t>() + tx;
                            const std::size_t localIdx = blockLin * pointsPerBlock + threadLin;

                            for (std::size_t field = 0; field < nFields; field++)
                            {
                                const std::size_t srcIdx = field * nNodes + virtualDeviceIndex * nPointsPerGPU + localIdx;
                                soa[field][idxGlobal] = fMom[srcIdx];
                            }
                        });
                });

            return soa;
        }

    }

}

#endif
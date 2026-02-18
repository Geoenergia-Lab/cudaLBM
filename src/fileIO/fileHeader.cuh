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
    fileHeader.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_FILEHEADER_CUH
#define __MBLBM_FILEHEADER_CUH

namespace LBM
{
    namespace fileIO
    {
        /**
         * @brief Reads a parameter value of type T from the provided lines based on the parameter name
         * @tparam T The type of the parameter to read (e.g., int, float)
         * @param[in] lines The lines of text to search for the parameter
         * @param[in] parameterName The name of the parameter to read
         * @return The value of the parameter converted to type T
         **/
        template <typename T>
        __host__ [[nodiscard]] inline T read(const words_t &lines, const name_t &parameterName)
        {
            return string::extractParameter<T>(lines, parameterName);
        }

        /**
         * @brief Reads a parameter of type T from the provided lines by comparing it to a particular string
         * @tparam T The type of the parameter to read (e.g., int, float)
         * @param[in] lines The lines of text to search for the parameter
         * @param[in] parameterName The name of the parameter to read
         * @param[in] trueString The string corresponding to a value of true
         * @return The value of the parameter converted to type T
         **/
        template <typename T>
        __host__ [[nodiscard]] T read(const words_t &lines, const name_t &parameterName, const name_t &trueString)
        {
            return (string::extractParameterLine(lines, parameterName) == trueString) ? static_cast<T>(true) : static_cast<T>(false);
        }

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

        class systemInformation
        {
        public:
            /**
             * @brief Constructs from text lines
             * @param[in] systemInfoLines Text lines read from the file
             **/
            __host__ [[nodiscard]] systemInformation(const words_t &systemInfoLines)
                : endianType_(read<endian::type>(systemInfoLines, "binaryType", "littleEndian")),
                  scalarSize_(read<std::size_t>(systemInfoLines, "scalarSize")){};

            /**
             * @brief Returns the endianness of the binary data
             * @return The endianness as a value of the endian::type enum
             **/
            __host__ [[nodiscard]] inline constexpr endian::type endianType() const noexcept
            {
                return endianType_;
            }

            /**
             * @brief Returns the size of the scalar values in bytes (e.g., 4 for float, 8 for double)
             * @return The size of scalar values in bytes
             **/
            __host__ [[nodiscard]] inline constexpr std::size_t scalarSize() const noexcept
            {
                return scalarSize_;
            }

        private:
            /**
             * @brief Endianness
             **/
            const endian::type endianType_;

            /**
             * @brief Size of the floating point type
             **/
            const std::size_t scalarSize_;
        };

        class meshPrimitive
        {
        public:
            /**
             * @brief Constructs a meshPrimitive object by parsing the provided lines of the lattice mesh block.
             * @param[in] meshLines The lines of the lattice mesh block to parse.
             **/
            __host__ [[nodiscard]] meshPrimitive(const words_t &meshLines)
                : nPoints_({read<label_t>(meshLines, "nx"), read<label_t>(meshLines, "ny"), read<label_t>(meshLines, "nz")}),
                  nDevices_({read<label_t>(meshLines, "nxGPUs"), read<label_t>(meshLines, "nyGPUs"), read<label_t>(meshLines, "nzGPUs")}){};

            /**
             * @brief Returns the number of lattice points in each direction as a blockLabel_t struct.
             * @return A blockLabel_t struct containing the number of lattice points in x, y, and z directions.
             **/
            __host__ [[nodiscard]] inline const blockLabel_t &nPoints() const noexcept
            {
                return nPoints_;
            }

            /**
             * @brief Returns the number of devices (GPUs) in each direction as a blockLabel_t struct.
             * @return A blockLabel_t struct containing the number of devices in x, y, and z directions.
             **/
            __host__ [[nodiscard]] inline const blockLabel_t &nDevices() const noexcept
            {
                return nDevices_;
            }

        private:
            /**
             * @brief The number of lattice points in each direction (x, y, z) as a blockLabel_t struct.
             **/
            const blockLabel_t nPoints_;

            /**
             * @brief The number of devices (GPUs) in each direction (x, y, z) as a blockLabel_t struct.
             **/
            const blockLabel_t nDevices_;
        };

        class fieldInformation
        {
        public:
            /**
             * @brief Constructs a fieldInformation object by parsing the provided lines of the field information block.
             * @param[in] fieldInfoLines The lines of the field information block to parse.
             **/
            __host__ [[nodiscard]] fieldInformation(const words_t &fieldInfoLines)
                : timeStep_(read<std::size_t>(fieldInfoLines, "timeStep")),
                  timeType_(read<time::type>(fieldInfoLines, "timeType", "instantaneous")),
                  nFields_(read<std::size_t>(fieldInfoLines, "nFields")),
                  fieldNames_(readFieldNames(fieldInfoLines, nFields_)){};

            /**
             * @brief Returns the time step of the saved fields.
             * @return The time step as a size_t.
             **/
            __host__ [[nodiscard]] std::size_t timeStep() const noexcept
            {
                return timeStep_;
            }

            /**
             * @brief Returns the time type (instantaneous or time average).
             * @return The time type as a value of the time::type enum.
             **/
            __host__ [[nodiscard]] time::type timeType() const noexcept
            {
                return timeType_;
            }

            /**
             * @brief Returns the number of fields.
             * @return The number of fields as a size_t.
             **/
            __host__ [[nodiscard]] std::size_t nFields() const noexcept
            {
                return nFields_;
            }

            /**
             * @brief Returns the field names as a vector of strings.
             * @return A vector containing the field names.
             **/
            __host__ [[nodiscard]] const std::vector<std::string> &fieldNames() const noexcept
            {
                return fieldNames_;
            }

        private:
            /**
             * @brief The time step of the saved fields as a size_t.
             **/
            const std::size_t timeStep_;

            /**
             * @brief The time type of the saved fields (instantaneous or time average) as a value of the time::type enum.
             **/
            const time::type timeType_;

            /**
             * @brief The number of fields as a size_t.
             **/
            const std::size_t nFields_;

            /**
             * @brief The field names as a vector of strings.
             **/
            const std::vector<std::string> fieldNames_;

            /**
             * @brief Reads the field names from the field information block.
             * @param[in] fieldInfoLines The lines of the field information block.
             * @return A vector containing the field names.
             **/
            __host__ [[nodiscard]] static words_t readFieldNames(const words_t &fieldInfoLines, const std::size_t N)
            {
                words_t B = string::extractBlock(fieldInfoLines, "fieldNames[" + std::to_string(N) + "]", 0);

                for (std::size_t i = 1; i < B.size() - 1; i++)
                {
                    B[i] = filestring_trim(B[i]);
                }

                return words_t(B.begin() + 1, B.end() - 2);
            }
        };

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
                    else if (line.find("scalarSize") != name_t::npos)
                    {
                        if (line.find("32") != name_t::npos)
                        {
                            scalarSize = 4;
                        }
                        else if (line.find("64") != name_t::npos)
                        {
                            scalarSize = 8;
                        }
                        else
                        {
                            throw std::runtime_error("Invalid scalarSize at line " + std::to_string(lineNumber));
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
                throw std::runtime_error("Missing or invalid scalarSize in systemInformation");
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

    }

}

#endif
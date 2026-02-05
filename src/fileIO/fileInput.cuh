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
        // Forward declarations
        struct FieldInformation;
        struct SystemInformation;
        struct FieldData;

        // Data structures for parsed sections
        struct FieldInformation
        {
            std::vector<std::string> fieldNames;
            std::size_t expectedFieldCount = 0;
            std::string timeType = "instantaneous"; // default
            std::size_t timeStep = 0;               // default
            std::size_t meanCount = 0;              // default, only used for timeAverage
        };

        struct SystemInformation
        {
            bool isLittleEndian = false;
            std::size_t scalarSize = 0;
        };

        struct FieldData
        {
            std::size_t nx = 0;
            std::size_t ny = 0;
            std::size_t nz = 0;
            std::size_t nVars = 0;
            std::size_t totalPoints = 0;
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
            const bool isLittleEndian;
            const std::size_t scalarSize;
            const std::size_t nx;
            const std::size_t ny;
            const std::size_t nz;
            const std::size_t nVars;
            const std::size_t dataStartPos;
            const std::vector<std::string> fieldNames;
            const std::string timeType; // "instantaneous" or "timeAverage"
            const std::size_t timeStep;
            const std::size_t meanCount; // only meaningful if timeType == "timeAverage"

            /**
             * @brief Constructor that parses the file and initializes all const members
             * @param[in] fileName The name of the file to parse
             * @throws std::runtime_error if the file cannot be read or is invalid
             **/
            __host__ explicit fieldFileHeader(const std::string &fileName)
                : isLittleEndian(initializeIsLittleEndian(fileName)),
                  scalarSize(initializeScalarSize(fileName)),
                  nx(initializeNx(fileName)),
                  ny(initializeNy(fileName)),
                  nz(initializeNz(fileName)),
                  nVars(initializeNVars(fileName)),
                  dataStartPos(initializeDataStartPos(fileName)),
                  fieldNames(initializeFieldNames(fileName)),
                  timeType(initializeTimeType(fileName)),
                  timeStep(initializeTimeStep(fileName)),
                  meanCount(initializeMeanCount(fileName))
            {
                // Additional validation can be done here if needed
                validateInternalConsistency();
            }

        private:
            // Helper functions to parse individual components
            __host__ [[nodiscard]] static bool initializeIsLittleEndian(const std::string &fileName);
            __host__ [[nodiscard]] static std::size_t initializeScalarSize(const std::string &fileName);
            __host__ [[nodiscard]] static std::size_t initializeNx(const std::string &fileName);
            __host__ [[nodiscard]] static std::size_t initializeNy(const std::string &fileName);
            __host__ [[nodiscard]] static std::size_t initializeNz(const std::string &fileName);
            __host__ [[nodiscard]] static std::size_t initializeNVars(const std::string &fileName);
            __host__ [[nodiscard]] static std::size_t initializeDataStartPos(const std::string &fileName);
            __host__ [[nodiscard]] static const std::vector<std::string> initializeFieldNames(const std::string &fileName);
            __host__ [[nodiscard]] static const std::string initializeTimeType(const std::string &fileName);
            __host__ [[nodiscard]] static std::size_t initializeTimeStep(const std::string &fileName);
            __host__ [[nodiscard]] static std::size_t initializeMeanCount(const std::string &fileName);

            // Common parsing helper
            __host__ [[nodiscard]] static std::tuple<std::vector<std::string>, std::size_t, std::size_t>
            readAndParseFile(const std::string &fileName);

            // Validation
            __host__ void validateInternalConsistency() const;
        };

        /**
         * @brief Trims leading/trailing whitespace and trailing semicolons from a string
         * @param[in] str The input string to trim
         * @return The trimmed string
         **/
        __host__ [[nodiscard]] const std::string filestring_trim(const std::string &str)
        {
            const std::size_t start = str.find_first_not_of(" \t\r\n");
            const std::size_t end = str.find_last_not_of(" \t\r\n;");
            return (start == std::string::npos) ? "" : str.substr(start, end - start + 1);
        }

        // Validate overall consistency
        __host__ void validateHeaderConsistency(
            const SystemInformation &systemInfo,
            const FieldInformation &fieldInfo,
            const FieldData &fieldData,
            const std::size_t dataStartPos,
            const std::size_t fileSize)
        {
            // Check field names match nVars
            if (fieldInfo.fieldNames.size() != fieldData.nVars)
            {
                throw std::runtime_error(
                    "Field names count (" + std::to_string(fieldInfo.fieldNames.size()) +
                    ") does not match nVars (" + std::to_string(fieldData.nVars) + ")");
            }

            // Additional validation for time-averaged fields
            if (fieldInfo.timeType == "timeAverage")
            {
                // For time-averaged fields, we might want to validate meanCount
                // For example, ensure it's not unreasonably large
                constexpr std::size_t MAX_REASONABLE_MEAN_COUNT = 1000000;
                if (fieldInfo.meanCount > MAX_REASONABLE_MEAN_COUNT)
                {
                    throw std::runtime_error(
                        "Unreasonably large meanCount: " + std::to_string(fieldInfo.meanCount));
                }

                // If meanCount is 0, it might be the start of averaging - that's ok
                // Could add a warning here if desired
            }

            // Check binary data size
            if (fieldData.totalPoints > std::numeric_limits<std::size_t>::max() / systemInfo.scalarSize)
            {
                throw std::runtime_error("Data size calculation would overflow");
            }

            const std::size_t expectedDataSize = fieldData.totalPoints * systemInfo.scalarSize;
            if (fileSize < dataStartPos + expectedDataSize)
            {
                throw std::runtime_error(
                    "Insufficient data in file. Expected " + std::to_string(expectedDataSize) +
                    " bytes from position " + std::to_string(dataStartPos) +
                    ", but file only has " + std::to_string(fileSize) + " bytes");
            }
        }

        // Parse system information section
        __host__ [[nodiscard]] const SystemInformation parseSystemInformation(const std::vector<std::string> &headerLines)
        {
            try
            {
                // Extract the systemInformation block
                const auto systemInfoBlock = LBM::string::extractBlock(headerLines, "systemInformation");

                // Remove the outer braces
                const auto innerLines = LBM::string::eraseBraces(systemInfoBlock);

                SystemInformation result;

                // Parse each line in the system information block
                for (const auto &line : innerLines)
                {
                    const auto trimmedLine = LBM::string::trim<string::TRIM_SEMICOLON>(line);
                    if (trimmedLine.empty())
                        continue;

                    // Split by whitespace to get key-value pairs
                    const auto tokens = LBM::string::split<' '>(trimmedLine, true);
                    if (tokens.size() < 2)
                        continue;

                    const auto &key = tokens[0];
                    const auto &value = tokens[1];

                    if (key == "binaryType")
                    {
                        result.isLittleEndian = (value == "littleEndian");
                    }
                    else if (key == "scalarType")
                    {
                        if (value.find("32") != std::string::npos)
                        {
                            result.scalarSize = 4;
                        }
                        else if (value.find("64") != std::string::npos)
                        {
                            result.scalarSize = 8;
                        }
                        else
                        {
                            throw std::runtime_error("Invalid scalarType: " + value);
                        }
                    }
                }

                if (result.scalarSize == 0)
                {
                    throw std::runtime_error("Missing or invalid scalarType in systemInformation");
                }

                return result;
            }
            catch (const std::runtime_error &e)
            {
                throw std::runtime_error("Error parsing systemInformation: " + std::string(e.what()));
            }
        }

        __host__ [[nodiscard]] const std::vector<std::string> extractFieldNamesFromBlock(
            const std::vector<std::string> &innerLines,
            const std::size_t fieldNamesStartLine)
        {
            std::vector<std::string> fieldNames;

            // We'll use the existing extractBlock utility
            // Create a sub-vector starting from the fieldNames line
            std::vector<std::string> remainingLines;

            // Convert fieldNamesStartLine to signed type for iterator arithmetic
            const auto startIdx = static_cast<std::ptrdiff_t>(fieldNamesStartLine);
            const auto endIdx = static_cast<std::ptrdiff_t>(innerLines.size());

            // Use vector's iterator arithmetic
            remainingLines.assign(
                innerLines.begin() + startIdx,
                innerLines.begin() + endIdx);

            // Extract the fieldNames block
            const auto fieldNamesBlock = LBM::string::extractBlock(remainingLines, "fieldNames");

            // Remove braces
            const auto innerFieldNames = LBM::string::eraseBraces(fieldNamesBlock);

            // Parse field names
            for (const auto &line : innerFieldNames)
            {
                const std::string trimmed = LBM::string::trim<string::TRIM_SEMICOLON>(line);
                if (!trimmed.empty())
                {
                    std::string fieldName = trimmed;
                    // Remove trailing semicolon
                    if (!fieldName.empty() && fieldName.back() == ';')
                    {
                        fieldName.pop_back();
                    }

                    fieldName = LBM::string::trim<string::TRIM_SEMICOLON>(fieldName);

                    if (!fieldName.empty())
                    {
                        // Check for duplicates
                        if (LBM::string::containsString(fieldNames, fieldName))
                        {
                            throw std::runtime_error("Duplicate field name: " + fieldName);
                        }
                        fieldNames.push_back(fieldName);
                    }
                }
            }

            return fieldNames;
        }

        // Parse field information section
        __host__ [[nodiscard]] const FieldInformation parseFieldInformation(
            const std::vector<std::string> &headerLines)
        {
            try
            {
                // Extract the fieldInformation block
                const auto fieldInfoBlock = LBM::string::extractBlock(headerLines, "fieldInformation");

                // Remove the outer braces
                const auto innerLines = LBM::string::eraseBraces(fieldInfoBlock);

                FieldInformation result;
                bool foundTimeType = false;
                bool foundMeanCount = false;

                // Parse key-value pairs in the fieldInformation block
                for (size_t i = 0; i < innerLines.size(); ++i)
                {
                    const auto trimmedLine = LBM::string::trim<string::TRIM_SEMICOLON>(innerLines[i]);
                    if (trimmedLine.empty())
                        continue;

                    // Check if this is the fieldNames declaration
                    if (trimmedLine.find("fieldNames[") != std::string::npos)
                    {
                        // Extract the count from fieldNames[10]
                        const auto startBracket = trimmedLine.find('[');
                        const auto endBracket = trimmedLine.find(']');
                        if (startBracket == std::string::npos || endBracket == std::string::npos)
                        {
                            throw std::runtime_error("Invalid fieldNames format");
                        }

                        const std::string countStr = trimmedLine.substr(
                            startBracket + 1,
                            endBracket - startBracket - 1);

                        try
                        {
                            result.expectedFieldCount = LBM::string::extractParameter<std::size_t>(countStr);
                            if (result.expectedFieldCount == 0)
                            {
                                throw std::runtime_error("Field names count cannot be zero");
                            }
                        }
                        catch (const std::exception &e)
                        {
                            throw std::runtime_error("Invalid fieldNames count: " + countStr);
                        }

                        // Extract field names from the nested block
                        result.fieldNames = extractFieldNamesFromBlock(innerLines, i);
                        break;
                    }
                    else
                    {
                        // Parse key-value pairs (timeStep, timeType, meanCount)
                        // Split by whitespace to get key and value
                        const auto tokens = LBM::string::split<' '>(trimmedLine, true);
                        if (tokens.size() >= 2)
                        {
                            std::string key = tokens[0];
                            std::string value = tokens[1];

                            // Remove trailing semicolon if present
                            if (!value.empty() && value.back() == ';')
                            {
                                value.pop_back();
                            }

                            if (key == "timeStep")
                            {
                                try
                                {
                                    result.timeStep = LBM::string::extractParameter<std::size_t>(value);
                                }
                                catch (const std::exception &e)
                                {
                                    throw std::runtime_error("Invalid timeStep: " + value);
                                }
                            }
                            else if (key == "timeType")
                            {
                                result.timeType = value;
                                foundTimeType = true;

                                // Validate timeType
                                if (value != "instantaneous" && value != "timeAverage")
                                {
                                    throw std::runtime_error("Invalid timeType: " + value +
                                                             ". Must be 'instantaneous' or 'timeAverage'");
                                }
                            }
                            else if (key == "meanCount")
                            {
                                try
                                {
                                    result.meanCount = LBM::string::extractParameter<std::size_t>(value);
                                    foundMeanCount = true;
                                }
                                catch (const std::exception &e)
                                {
                                    throw std::runtime_error("Invalid meanCount: " + value);
                                }
                            }
                        }
                    }
                }

                // Validation
                if (!foundTimeType)
                {
                    throw std::runtime_error("Missing timeType in fieldInformation");
                }

                if (result.expectedFieldCount == 0)
                {
                    throw std::runtime_error("No fieldNames found in fieldInformation");
                }

                // Validate field names count
                if (result.fieldNames.size() != result.expectedFieldCount)
                {
                    throw std::runtime_error(
                        "Field names count (" + std::to_string(result.fieldNames.size()) +
                        ") does not match declared count (" + std::to_string(result.expectedFieldCount) + ")");
                }

                // Validate meanCount based on timeType
                if (result.timeType == "timeAverage" && !foundMeanCount)
                {
                    throw std::runtime_error("timeAverage specified but meanCount not found");
                }

                if (result.timeType == "instantaneous" && foundMeanCount)
                {
                    // Warn or throw? Let's be strict and throw
                    throw std::runtime_error("meanCount found for instantaneous field. "
                                             "meanCount should only be used with timeAverage fields.");
                }

                return result;
            }
            catch (const std::runtime_error &e)
            {
                throw std::runtime_error("Error parsing fieldInformation: " + std::string(e.what()));
            }
        }

        // Parse field data section
        __host__ [[nodiscard]] const FieldData parseFieldData(const std::vector<std::string> &headerLines)
        {
            try
            {
                // Extract the fieldData block
                const auto fieldDataBlock = LBM::string::extractBlock(headerLines, "fieldData");

                FieldData result;

                // Look for the field[...] line
                for (const auto &line : fieldDataBlock)
                {
                    const auto trimmedLine = LBM::string::trim<string::UNTRIMMED_SEMICOLON>(line);

                    if (trimmedLine.find("field[") != std::string::npos)
                    {
                        // Extract dimensions from pattern: field[total][nVars][nz][ny][nx]
                        std::vector<std::size_t> dims;
                        std::size_t pos = 0;

                        while ((pos = trimmedLine.find('[', pos)) != std::string::npos)
                        {
                            const std::size_t end = trimmedLine.find(']', pos);
                            if (end == std::string::npos)
                            {
                                throw std::runtime_error("Unclosed bracket in field dimensions");
                            }

                            try
                            {
                                const std::string dimStr = trimmedLine.substr(pos + 1, end - pos - 1);
                                const unsigned long long dimValue = std::stoull(dimStr);

                                if (dimValue > std::numeric_limits<std::size_t>::max())
                                {
                                    throw std::runtime_error("Dimension value too large");
                                }

                                dims.push_back(static_cast<std::size_t>(dimValue));
                            }
                            catch (const std::exception &e)
                            {
                                throw std::runtime_error("Invalid dimension format: " + std::string(e.what()));
                            }
                            pos = end + 1;
                        }

                        if (dims.size() < 5)
                        {
                            throw std::runtime_error("Invalid field dimensions");
                        }

                        // Note: The original code had dims[1] as nVars, dims[2] as nz, etc.
                        // According to the pattern field[total][nVars][nz][ny][nx]
                        result.totalPoints = dims[0];
                        result.nVars = dims[1];
                        result.nz = dims[2];
                        result.ny = dims[3];
                        result.nx = dims[4];

                        // Validate dimensions
                        if (result.nx == 0 || result.ny == 0 || result.nz == 0 || result.nVars == 0)
                        {
                            throw std::runtime_error("Invalid grid dimensions: nx, ny, nz, nVars cannot be zero");
                        }

                        // Check for potential overflow
                        if (result.nx > std::numeric_limits<std::size_t>::max() / result.ny / result.nz / result.nVars)
                        {
                            throw std::runtime_error("Dimension product would overflow");
                        }

                        // Verify consistency
                        if (result.totalPoints != result.nx * result.ny * result.nz * result.nVars)
                        {
                            throw std::runtime_error(
                                "Dimension mismatch: total points (" + std::to_string(result.totalPoints) +
                                ") != nx * ny * nz * nVars (" + std::to_string(result.nx * result.ny * result.nz * result.nVars) + ")");
                        }

                        return result;
                    }
                }

                throw std::runtime_error("No field dimensions found in fieldData");
            }
            catch (const std::runtime_error &e)
            {
                throw std::runtime_error("Error parsing fieldData: " + std::string(e.what()));
            }
        }

        // Helper function to parse all sections from header lines
        __host__ [[nodiscard]] const std::tuple<SystemInformation, FieldInformation, FieldData, std::size_t, std::size_t>
        parseHeaderSections(const std::vector<std::string> &headerLines, const std::size_t dataStartPos, const std::size_t fileSize)
        {
            // Parse system information
            const auto systemInfo = parseSystemInformation(headerLines);

            // Parse field information
            const auto fieldInfo = parseFieldInformation(headerLines);

            // Parse field data
            const auto fieldData = parseFieldData(headerLines);

            return std::make_tuple(systemInfo, fieldInfo, fieldData, dataStartPos, fileSize);
        }

        // Helper function to read and parse file (used by fieldFileHeader constructor)
        __host__ [[nodiscard]] std::tuple<std::vector<std::string>, std::size_t, std::size_t>
        fieldFileHeader::readAndParseFile(const std::string &fileName)
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

            if (fileSizePos < 0)
            {
                throw std::runtime_error("Invalid file size (negative)");
            }

            const std::size_t fileSize = static_cast<std::size_t>(fileSizePos);
            in.seekg(0, std::ios::beg);

            // Read the header text portion
            std::vector<std::string> headerLines;
            std::string line;
            std::size_t lineNumber = 0;
            bool foundFieldDataStart = false;
            std::size_t dataStartPos = 0;

            // Read until we find the start of binary data
            while (std::getline(in, line))
            {
                lineNumber++;

                // Clean the line
                const std::string cleanedLine = LBM::string::trim<string::UNTRIMMED_SEMICOLON>(line);

                // Check if we've reached the start of binary data
                if (foundFieldDataStart && cleanedLine == "{")
                {
                    // This is the opening brace before binary data
                    const auto pos = in.tellg();
                    if (pos == -1)
                    {
                        throw std::runtime_error("Cannot determine binary data start position");
                    }
                    if (pos < 0)
                    {
                        throw std::runtime_error("Invalid file position (negative)");
                    }
                    dataStartPos = static_cast<std::size_t>(pos);
                    break;
                }

                // Check if we're about to enter binary data
                if (!foundFieldDataStart && cleanedLine.find("field[") != std::string::npos)
                {
                    foundFieldDataStart = true;
                }

                headerLines.push_back(line);

                // Safety check
                if (lineNumber > 1000)
                {
                    throw std::runtime_error("Header too large or malformed");
                }
            }

            if (!foundFieldDataStart)
            {
                throw std::runtime_error("Could not find field data section");
            }

            return std::make_tuple(headerLines, dataStartPos, fileSize);
        }

        // Individual initialization functions for fieldFileHeader constructor
        __host__ [[nodiscard]] bool fieldFileHeader::initializeIsLittleEndian(const std::string &fileName)
        {
            auto [headerLines, dataStartPos, fileSize] = readAndParseFile(fileName);
            const auto systemInfo = parseSystemInformation(headerLines);
            return systemInfo.isLittleEndian;
        }

        __host__ [[nodiscard]] std::size_t fieldFileHeader::initializeScalarSize(const std::string &fileName)
        {
            auto [headerLines, dataStartPos, fileSize] = readAndParseFile(fileName);
            const auto systemInfo = parseSystemInformation(headerLines);
            return systemInfo.scalarSize;
        }

        __host__ [[nodiscard]] std::size_t fieldFileHeader::initializeNx(const std::string &fileName)
        {
            auto [headerLines, dataStartPos, fileSize] = readAndParseFile(fileName);
            const auto fieldData = parseFieldData(headerLines);
            return fieldData.nx;
        }

        __host__ [[nodiscard]] std::size_t fieldFileHeader::initializeNy(const std::string &fileName)
        {
            auto [headerLines, dataStartPos, fileSize] = readAndParseFile(fileName);
            const auto fieldData = parseFieldData(headerLines);
            return fieldData.ny;
        }

        __host__ [[nodiscard]] std::size_t fieldFileHeader::initializeNz(const std::string &fileName)
        {
            auto [headerLines, dataStartPos, fileSize] = readAndParseFile(fileName);
            const auto fieldData = parseFieldData(headerLines);
            return fieldData.nz;
        }

        __host__ [[nodiscard]] std::size_t fieldFileHeader::initializeNVars(const std::string &fileName)
        {
            auto [headerLines, dataStartPos, fileSize] = readAndParseFile(fileName);
            const auto fieldData = parseFieldData(headerLines);
            return fieldData.nVars;
        }

        __host__ [[nodiscard]] std::size_t fieldFileHeader::initializeDataStartPos(const std::string &fileName)
        {
            auto [headerLines, dataStartPos, fileSize] = readAndParseFile(fileName);
            return dataStartPos;
        }

        __host__ [[nodiscard]] const std::vector<std::string> fieldFileHeader::initializeFieldNames(const std::string &fileName)
        {
            auto [headerLines, dataStartPos, fileSize] = readAndParseFile(fileName);
            const auto fieldInfo = parseFieldInformation(headerLines);
            return fieldInfo.fieldNames;
        }

        __host__ [[nodiscard]] const std::string fieldFileHeader::initializeTimeType(const std::string &fileName)
        {
            auto [headerLines, dataStartPos, fileSize] = readAndParseFile(fileName);
            const auto fieldInfo = parseFieldInformation(headerLines);
            return fieldInfo.timeType;
        }

        __host__ [[nodiscard]] std::size_t fieldFileHeader::initializeTimeStep(const std::string &fileName)
        {
            auto [headerLines, dataStartPos, fileSize] = readAndParseFile(fileName);
            const auto fieldInfo = parseFieldInformation(headerLines);
            return fieldInfo.timeStep;
        }

        __host__ [[nodiscard]] std::size_t fieldFileHeader::initializeMeanCount(const std::string &fileName)
        {
            auto [headerLines, dataStartPos, fileSize] = readAndParseFile(fileName);
            const auto fieldInfo = parseFieldInformation(headerLines);
            return fieldInfo.meanCount;
        }

        __host__ void fieldFileHeader::validateInternalConsistency() const
        {
            // Check field names count matches nVars
            if (fieldNames.size() != nVars)
            {
                throw std::runtime_error(
                    "Field names count (" + std::to_string(fieldNames.size()) +
                    ") does not match nVars (" + std::to_string(nVars) + ")");
            }

            // Validate timeType
            if (timeType != "instantaneous" && timeType != "timeAverage")
            {
                throw std::runtime_error("Invalid timeType: " + timeType);
            }

            // Validate meanCount based on timeType
            if (timeType == "timeAverage")
            {
                constexpr std::size_t MAX_REASONABLE_MEAN_COUNT = 1000000;
                if (meanCount > MAX_REASONABLE_MEAN_COUNT)
                {
                    throw std::runtime_error(
                        "Unreasonably large meanCount: " + std::to_string(meanCount));
                }
            }
            else if (timeType == "instantaneous" && meanCount != 0)
            {
                throw std::runtime_error("meanCount should be 0 for instantaneous fields");
            }

            // Validate dimensions
            if (nx == 0 || ny == 0 || nz == 0 || nVars == 0)
            {
                throw std::runtime_error("Invalid grid dimensions: nx, ny, nz, nVars cannot be zero");
            }

            // Check for potential overflow
            if (nx > std::numeric_limits<std::size_t>::max() / ny / nz / nVars)
            {
                throw std::runtime_error("Dimension product would overflow");
            }

            // Verify consistency of total points
            const std::size_t calculatedTotalPoints = nx * ny * nz * nVars;
            // Note: We don't have totalPoints as a member anymore, but we can still check nx*ny*nz*nVars is reasonable
            (void)calculatedTotalPoints; // Suppress unused variable warning
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
        __host__ [[nodiscard]] const std::vector<T> readFieldFile(const std::string &fileName)
        {
            static_assert(std::is_floating_point_v<T>, "T must be floating point");
            static_assert(std::endian::native == std::endian::little || std::endian::native == std::endian::big, "System must be little or big endian");

            // Parse header metadata using the new constructor
            const fieldFileHeader header(fileName);

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
        __host__ [[nodiscard]] const std::vector<T> readFieldByName(const std::string &fileName, const std::string &fieldName)
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

            // Parse header using the new constructor
            const fieldFileHeader header(fileName);

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
                std::string availableFields;
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

            // Calculate field index and data position
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

            // Safety check for the size of fMom and nPoints
            if (fMom.size() % nNodes != 0)
            {
                throw std::invalid_argument("fMom size (" + std::to_string(fMom.size()) + ") is not divisible by mesh points (" + std::to_string(nNodes) + ")");
            }

            const std::size_t nFields = fMom.size() / nNodes;

            std::vector<std::vector<T>> soa(nFields, std::vector<scalar_t>(nNodes, 0));

#ifdef MULTI_GPU
            static_assert(false, "deinterleaveAoS not implemented for multi GPU yet");
#else
            grid_for(
                mesh.nxBlocks(), mesh.nyBlocks(), mesh.nzBlocks(),
                [&](const label_t bx, const label_t by, const label_t bz,
                    const label_t tx, const label_t ty, const label_t tz)
                {
                    const label_t x = (bx * block::nx()) + tx;
                    const label_t y = (by * block::ny()) + ty;
                    const label_t z = (bz * block::nz()) + tz;

                    // MODIFY FOR MULTI GPU: idx must be multi GPU aware
                    const label_t idxGlobal = host::idxScalarGlobal(x, y, z, mesh.nx(), mesh.ny());
                    const label_t idx = host::idx(tx, ty, tz, bx, by, bz, mesh);

                    for (label_t field = 0; field < nFields; field++)
                    {
                        soa[field][idxGlobal] = fMom[idx + (field * mesh.nPoints())];
                    }
                });
#endif

            return soa;
        }

    }

}

#endif
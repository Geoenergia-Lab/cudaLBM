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
    Top-level header for the file IO operations

Namespace
    LBM::fileIO

SourceFiles
    fileIO.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_FILEIO_CUH
#define __MBLBM_FILEIO_CUH

#include "../LBMIncludes.cuh"
#include "../typedefs/typedefs.cuh"
#include "fileSystem.cuh"
#include "../memory/memory.cuh"

namespace LBM
{
    namespace fileIO
    {
        /**
         * @brief Safely converts an integer of type T to a std::streamsize
         * @param[in] size The size to convert
         **/
        template <typename T>
        __host__ [[nodiscard]] std::streamsize to_streamsize(const T size)
        {
            types::assertions::validate<T>();

            static_assert(std::is_integral_v<T>, "Conversion to std::streamsize must be from an integral type");

            if (size > static_cast<T>(std::numeric_limits<std::streamsize>::max()))
            {
                throw std::runtime_error("Data size " + std::to_string(size) + " exceeds maximum stream size");
                return 0;
            }
            else
            {
                return static_cast<std::streamsize>(size);
            }
        }

        /**
         * @brief Validates if a string represents a valid integer
         * @param[in] intStr String to validate
         * @return True if string is non-empty and contains only digits
         **/
        __host__ [[nodiscard]] inline bool isValidInteger(const name_t &intStr) noexcept
        {
            return (!intStr.empty() || string::isAllDigits(intStr));
        }

        /**
         * @brief Checks if directory contains indexed .LBMBin files with given prefix
         * @param[in] fileName Case name prefix to search for
         * @return True if matching files found, false otherwise
         *
         * Searches current directory for files with pattern: {fileName}_{number}.LBMBin
         * where {number} consists of only digits.
         **/
        __host__ [[nodiscard]] bool hasIndexedFiles(const name_t &fileName)
        {
            const std::filesystem::path currentDir = std::filesystem::current_path();
            const name_t prefix = fileName + "_";

            for (const auto &entry : std::filesystem::directory_iterator(currentDir))
            {
                if (!entry.is_regular_file())
                {
                    continue;
                }

                const auto &filePath = entry.path();
                if (filePath.extension() != ".LBMBin")
                {
                    continue;
                }

                const name_t stem = filePath.stem().string();
                if (stem.size() <= prefix.size() || stem.substr(0, prefix.size()) != prefix)
                {
                    continue;
                }

                const name_t num_str = stem.substr(prefix.size());
                if (num_str.empty())
                {
                    continue;
                }

                if (string::isAllDigits(num_str))
                {
                    return true;
                }
            }

            return false;
        }

        /**
         * @brief Extracts time indices from indexed files in current directory
         * @param[in] fileName Case name prefix to search for
         * @return Sorted vector of time indices found
         * @throws std::runtime_error if no valid files found
         *
         * Parses files with pattern: {fileName}_{number}.LBMBin and extracts
         * the numeric portion as time indices.
         **/
        __host__ [[nodiscard]] const std::vector<host::label_t> timeIndices(const name_t &fileName)
        {
            std::vector<host::label_t> indices;
            const std::filesystem::path currentDir = std::filesystem::current_path();
            const name_t prefix = fileName + "_";

            for (const auto &entry : std::filesystem::directory_iterator(currentDir))
            {
                if (!entry.is_regular_file())
                {
                    continue;
                }

                const std::filesystem::path &filePath = entry.path();
                if (filePath.extension() != ".LBMBin")
                {
                    continue;
                }

                const name_t stem = filePath.stem().string();
                if (stem.size() <= prefix.size() || stem.substr(0, prefix.size()) != prefix)
                {
                    continue;
                }

                const name_t num_str = stem.substr(prefix.size());
                if (!isValidInteger(num_str))
                {
                    continue;
                }

                try
                {
                    indices.push_back(LBM::string::extractParameter<host::label_t>(num_str));
                }
                catch (...)
                {
                    continue;
                }
            }

            // Check that the indices are empty - if they are, it means no valid files were found
            if (indices.empty())
            {
                throw std::runtime_error("No matching files found with prefix " + fileName + " and .LBMBin extension");
            }

            std::sort(indices.begin(), indices.end());

            return indices;
        }

        /**
         * @brief Gets the latest time index from available files
         * @param[in] fileName Case name prefix to search for
         * @return Highest time index found, or 0 if no files found
         **/
        __host__ [[nodiscard]] host::label_t latestTime(const name_t &fileName)
        {
            if (hasIndexedFiles(fileName))
            {
                const std::vector<host::label_t> indices = timeIndices(fileName);
                return indices[indices.size() - 1];
            }
            else
            {
                return 0;
            }
        }

        /**
         * @brief Determines starting index for field conversion loop
         * @tparam PC Program control type
         * @param[in] programCtrl The program control object
         * @param[in] isLatestTime Flag indicating whether to start from latest time
         * @return Starting index (0 for earliest, last index for latest)
         **/
        __host__ [[nodiscard]] host::label_t getStartIndex(const name_t &fileNamePrefix, const bool isLatestTime)
        {
            const std::vector<host::label_t> fileNameIndices = fileIO::timeIndices(fileNamePrefix);

            return isLatestTime ? static_cast<host::label_t>(fileNameIndices.size() - 1) : 0;
        }

        template <class ProgramControl>
        __host__ [[nodiscard]] host::label_t getStartIndex(const ProgramControl &programCtrl, const bool isLatestTime)
        {
            return getStartIndex(programCtrl.caseName(), isLatestTime);
        }

        /**
         * @brief Determines starting index based on program control settings
         * @tparam PC Program control type
         * @param[in] programCtrl The program control object
         * @return Starting index determined by command line arguments
         **/
        template <class ProgramControl>
        __host__ [[nodiscard]] host::label_t getStartIndex(const ProgramControl &programCtrl)
        {
            return getStartIndex(programCtrl, programCtrl.input().isArgPresent("-latestTime"));
        }

        template <class ProgramControl>
        __host__ [[nodiscard]] host::label_t getStartIndex(const name_t &fileNamePrefix, const ProgramControl &programCtrl)
        {
            return getStartIndex(fileNamePrefix, programCtrl.input().isArgPresent("-latestTime"));
        }
    }
}

#include "fileOutput.cuh"
#include "fileHeader.cuh"
#include "fileInput.cuh"

#endif
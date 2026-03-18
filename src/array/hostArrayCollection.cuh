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
    A templated class for allocating collections of arrays on the CPU

Namespace
    LBM::host

SourceFiles
    hostArrayCollection.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_HOSTARRAYCOLLECTION_CUH
#define __MBLBM_HOSTARRAYCOLLECTION_CUH

namespace LBM
{
    namespace host
    {
        /**
         * @class arrayCollection
         * @brief Templated container for multiple field arrays with flexible initialization
         * @tparam T Data type of array elements
         * @tparam cType Constructor type specification
         **/
        template <typename T, const ctorType::type cType>
        class arrayCollection
        {
        public:
            /**
             * @brief Construct from program control and variable names
             * @param[in] programCtrl The program control object
             * @param[in] varNames Names of variables to include in collection
             * @param[in] mesh The lattice mesh
             **/
            __host__ [[nodiscard]] arrayCollection(const programControl &programCtrl, const words_t &varNames, const host::latticeMesh &mesh)
                : arr_(initialiseVector(programCtrl, mesh)),
                  varNames_(varNames) {}

            /**
             * @brief Construct from specific time index
             * @param[in] programCtrl The program control object
             * @param[in] varNames Names of variables to include
             * @param[in] timeIndex Specific time index to read from
             **/
            __host__ [[nodiscard]] arrayCollection(
                const programControl &programCtrl,
                const words_t &varNames,
                const host::label_t timeIndex)
                : arr_(initialiseVector(programCtrl, timeIndex)),
                  varNames_(varNames) {}

            /**
             * @brief Construct from latest available time
             * @param[in] programCtrl The program control object
             * @param[in] varNames Names of variables to include
             **/
            __host__ [[nodiscard]] arrayCollection(
                const programControl &programCtrl,
                const words_t &varNames)
                : arr_(initialiseVector(programCtrl)),
                  varNames_(varNames) {}

            /**
             * @brief Construct from a file prefix
             * @param[in] fileNamePrefix The prefix of the file
             * @param[in] varNames Names of variables to include
             * @param[in] timeIndex Specific time index to read from
             **/
            __host__ [[nodiscard]] arrayCollection(
                const name_t &fileNamePrefix,
                const words_t &varNames,
                const host::label_t timeIndex)
                : arr_(initialiseVector(fileNamePrefix, timeIndex)),
                  varNames_(varNames) {}

            /**
             * @brief Destructor for the host arrayCollection class
             **/
            ~arrayCollection() {}

            /**
             * @brief Get read-only access to underlying data
             * @return Const reference to data vector
             **/
            __host__ [[nodiscard]] inline constexpr const std::vector<T> &arr() const noexcept
            {
                return arr_;
            }

            /**
             * @brief Get variable names in collection
             * @return Const reference to variable names vector
             **/
            __host__ [[nodiscard]] inline const words_t &varNames() const noexcept
            {
                return varNames_;
            }

        private:
            /**
             * @brief The underlying std::vector
             **/
            const std::vector<T> arr_;

            /**
             * @brief Names of the solution variables
             **/
            const words_t varNames_;

            /**
             * @brief Initialize vector from mesh dimensions
             * @param[in] programCtrl The program control object
             * @param[in] mesh The lattice mesh
             * @return Initialized data vector
             * @throws std::runtime_error if indexed files not found
             **/
            __host__ [[nodiscard]] const std::vector<T> initialiseVector(const programControl &programCtrl, const host::latticeMesh &mesh) const
            {
                static_assert(cType == ctorType::MUST_READ, "Invalid constructor type");

                // Get the latest time step
                if (!fileIO::hasIndexedFiles(programCtrl.caseName()))
                {
                    throw std::runtime_error("Did not find indexed case files");
                }

                const name_t fileName = programCtrl.caseName() + "_" + std::to_string(fileIO::latestTime(programCtrl.caseName())) + ".LBMBin";
                return fileIO::readFieldFile<T>(fileName);
            }

            __host__ [[nodiscard]] const std::vector<T> initialiseVector(const name_t &fileNamePrefix, const host::label_t timeIndex) const
            {
                static_assert(cType == ctorType::MUST_READ, "Invalid constructor type");

                // Get the latest time step
                if (!fileIO::hasIndexedFiles(fileNamePrefix))
                {
                    throw std::runtime_error("Did not find indexed case files");
                }
                const name_t fileName = fileNamePrefix + "_" + std::to_string(fileIO::timeIndices(fileNamePrefix)[timeIndex]) + ".LBMBin";
                return fileIO::readFieldFile<T>(fileName);
            }

            /**
             * @brief Initialize vector from specific time index
             * @param[in] programCtrl The program control object
             * @param[in] timeIndex Time index to read from
             * @return Initialized data vector
             * @throws std::runtime_error if indexed files not found
             **/
            __host__ [[nodiscard]] const std::vector<T> initialiseVector(const programControl &programCtrl, const host::label_t timeIndex) const
            {
                static_assert(cType == ctorType::MUST_READ, "Invalid constructor type");

                // Get the correct time index
                if (fileIO::hasIndexedFiles(programCtrl.caseName()))
                {
                    const name_t fileName = programCtrl.caseName() + "_" + std::to_string(fileIO::timeIndices(programCtrl.caseName())[timeIndex]) + ".LBMBin";
                    return fileIO::readFieldFile<T>(fileName);
                }
                else
                {
                    throw std::runtime_error("Did not find indexed case files");
                }
            }

            /**
             * @brief Initialize vector from latest time
             * @param[in] programCtrl The program control object
             * @return Initialized data vector
             **/
            __host__ [[nodiscard]] const std::vector<T> initialiseVector(const programControl &programCtrl) const
            {
                return initialiseVector(programCtrl, fileIO::getStartIndex(programCtrl, true));
            }
        };
    }
}

#endif

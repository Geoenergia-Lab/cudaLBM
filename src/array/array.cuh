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
    Top-level header file for the array class

Namespace
    LBM

SourceFiles
    array.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_ARRAY_CUH
#define __MBLBM_ARRAY_CUH

#include "../LBMIncludes.cuh"
#include "../typedefs/typedefs.cuh"

#include "threadArray.cuh"

#include "../latticeMesh/latticeMesh.cuh"
#include "../fileIO/fileIO.cuh"
#include "../velocitySet/velocitySet.cuh"

#include "../boundaryConditions/normalVector.cuh"
#include "../boundaryConditions/boundaryValue.cuh"
#include "../boundaryConditions/boundaryRegion.cuh"
#include "../boundaryConditions/boundaryFields.cuh"

namespace LBM
{
    namespace host
    {
        /**
         * @brief Type of memory allocation on the host:
         * The memory is either pageable or pinned
         **/
        typedef enum Enum : bool
        {
            PAGED = 0,
            PINNED = 1
        } mallocType;
    }

    namespace field
    {
        /**
         * @brief Type of field to be allocated
         * @note The skeleton type contains only a pointer;
         * FULL_FIELD contains a pointer, name and a reference to the mesh
         **/
        typedef enum Enum : bool
        {
            SKELETON = 0,
            FULL_FIELD = 1
        } type;
    }

    /**
     * @brief Constructor read types
     * @note Has to be enumerated because there are only so many possible read configurations
     **/
    namespace ctorType
    {
        typedef enum Enum : int
        {
            NO_READ = 0,
            MUST_READ = 1,
            READ_IF_PRESENT = 2
        } type;
    }

    /**
     * Reads the first N lines from a file.
     * @param filename Path to the file.
     * @param n Number of lines to read (non‑negative).
     * @return A vector containing the first N lines (or fewer if the file ends).
     */
    __host__ [[nodiscard]] const std::vector<std::string> read_first_n_lines(const std::string &filename, const std::size_t n)
    {
        std::vector<std::string> lines;
        if (n <= 0)
        {
            return lines;
        } // nothing to read

        std::ifstream file(filename);
        if (!file.is_open())
        {
            std::cerr << "Error: Could not open file " << filename << std::endl;
            return lines; // return empty vector
        }

        std::string line;
        std::size_t count = 0;
        while (count < n && std::getline(file, line))
        {
            lines.push_back(line);
            ++count;
        }

        file.close();
        return lines;
    }

    __host__ [[nodiscard]] label_t initialiseMeanCount(const programControl &programCtrl)
    {
        if (fileIO::hasIndexedFiles(programCtrl.caseName()))
        {
            const name_t fileName = programCtrl.caseName() + "_" + std::to_string(fileIO::latestTime(programCtrl.caseName())) + ".LBMBin";

            const words_t lines = read_first_n_lines(fileName, 50);

            const fileIO::fieldInformation fieldInfo(string::extractBlock(lines, "fieldInformation", 0));

            return static_cast<label_t>(fieldInfo.meanCount());
        }
        else
        {
            return 0;
        }
    }
}

#include "hostArray.cuh"
#include "deviceArray.cuh"
#include "hostArrayCollection.cuh"

#endif

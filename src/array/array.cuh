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
    Top-level header file containing relevant includes and definitions for the
    array classes used in cudaLBM. The array classes are designed to manage
    field data on both the host and device, with specializations for different
    types of fields (block halos and full fields) and memory allocation
    strategies (pinned and pageable). This file also includes utility functions
    for reading field data from files and initializing the mean counter for
    time-averaged fields.

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
     * @brief Counts lines before the first occurrence of a target line.
     * @param[in] file Input file stream (position advanced).
     * @param[in] target The line content that stops counting (excluded).
     * @return Number of lines read before target; if target not found, returns total lines.
     **/
    __host__ [[nodiscard]] host::label_t line_count(std::ifstream &file, const name_t &target)
    {
        name_t line;
        host::label_t result = 0;
        // bool found = false;

        while (std::getline(file, line))
        {
            if (line == target)
            {
                // found = true;
                break;
            }
            ++result;
        }

        return result;
    }

    /**
     * @brief Reads a file line by line and returns a vector of all lines that appear
     * before the first line exactly equal to the target string.
     * If the target is not found, all lines from the file are returned.
     *
     * The function performs two passes:
     * 1. Count how many lines precede the target (or the whole file if target absent).
     * 2. Reserve that many slots in the vector and read the lines again, storing them.
     *
     * @param[in] filename Path to the file.
     * @param[in] target The exact line content at which to stop reading (not included).
     * @return Vector of strings containing the lines before the target.
     **/
    __host__ [[nodiscard]] const words_t read_until(const name_t &filename, const name_t &target)
    {
        std::ifstream file(filename);
        if (!file.is_open())
        {
            return {}; // return empty vector on open failure
        }

        // Count lines before target
        // If target not found, lineCount already holds total lines in file.
        const host::label_t lineCount = line_count(file, target);

        // Read and store exactly lineCount lines
        file.clear();                 // clear EOF and error flags
        file.seekg(0, std::ios::beg); // rewind to beginning

        name_t line;
        words_t lines;
        lines.reserve(lineCount); // allocate once

        for (host::label_t i = 0; i < lineCount; i++)
        {
            std::getline(file, line);
            lines.push_back(std::move(line));
        }

        return lines;
    }

    /**
     * @brief Initialises the mean counter from a file.
     * @param[in] programCtrl The program control object
     * @returns The mean counter as a device::label_t.
     **/
    __host__ [[nodiscard]] device::label_t initialiseMeanCount(const programControl &programCtrl)
    {
        if (fileIO::hasIndexedFiles(programCtrl.caseName()))
        {
            const name_t fileName = programCtrl.caseName() + "_" + std::to_string(fileIO::latestTime(programCtrl.caseName())) + ".LBMBin";

            const words_t lines = read_until(fileName, "fieldData");

            const fileIO::fieldInformation fieldInfo(string::extractBlock(lines, "fieldInformation", 0));

            return static_cast<device::label_t>(fieldInfo.meanCount());
        }
        else
        {
            return 0;
        }
    }
}

#include "host/array.cuh"
#include "device/array.cuh"
#include "hostArrayCollection.cuh"

#endif

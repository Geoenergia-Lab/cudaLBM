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
    Implementation of writing solution variables encoded in binary format

Namespace
    LBM::fileIO

SourceFiles
    fileOutput.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_FILEOUTPUT_CUH
#define __MBLBM_FILEOUTPUT_CUH

namespace LBM
{
    namespace fileIO
    {
        // Get the time type string
        template <const time::type TimeType>
        __host__ [[nodiscard]] const std::string timeTypeString() noexcept
        {
            static_assert((TimeType == time::instantaneous || (TimeType == time::timeAverage)), "Time type must be either instantaneous or timeAverage");

            if constexpr (TimeType == time::instantaneous)
            {
                return "instantaneous";
            }

            if constexpr (TimeType == time::timeAverage)
            {
                return "timeAverage";
            }
        }

        /**
         * @brief Implementation of the writing of the binary file
         * @param fileName Name of the file to be written
         * @param mesh The mesh
         * @param varNames The names of the solution variables
         * @param fields The solution variables encoded in interleaved AoS format
         * @param timeStep The current time step
         **/
        template <const time::type TimeType, class LatticeMesh, typename T>
        __host__ void writeFile(
            const std::string &fileName,
            const LatticeMesh &mesh,
            const std::vector<std::string> &varNames,
            const T *const ptrRestrict fields,
            const label_t timeStep)
        {
            static_assert(std::is_floating_point<T>::value, "T must be floating point");

            static_assert(std::endian::native == std::endian::little | std::endian::native == std::endian::big, "File system must be either little or big endian");

            static_assert(sizeof(T) == 4 | sizeof(T) == 8, "Error writing file: T must be either 32 or 64 bit");

            static_assert((TimeType == time::instantaneous || (TimeType == time::timeAverage)), "Time type must be either instantaneous or timeAverage");

            const std::size_t nVars = varNames.size();
            const std::size_t nPoints = static_cast<std::size_t>(mesh.nx()) * static_cast<std::size_t>(mesh.ny()) * static_cast<std::size_t>(mesh.nz());
            const std::size_t expectedSize = nPoints * nVars;

            // Check if there is enough disk space to store the file
            {
                const std::size_t expectedDiskUsage = expectedSize * sizeof(T);
                if (!fileSystem::hasEnoughSpace(expectedDiskUsage))
                {
                    const label_t availableSpace = fileSystem::availableDiskSpace();
                    throw std::runtime_error("Insufficient disk space to write file " + fileName + "\nRequired: " + std::to_string(expectedDiskUsage) + "\nAvailable: " + std::to_string(availableSpace));
                }
            }

            std::ofstream out(fileName, std::ios::binary);
            if (!out)
            {
                throw std::runtime_error("Cannot open file: " + fileName);
            }

            // Write the system information: binary endianness
            out << "systemInformation" << std::endl;
            out << "{" << std::endl;
            if constexpr (std::endian::native == std::endian::little)
            {
                out << "\tbinaryType\tlittleEndian;" << std::endl;
            }
            else if constexpr (std::endian::native == std::endian::big)
            {
                out << "\tbinaryType\tbigEndian;" << std::endl;
            }
            out << std::endl;
            if constexpr (sizeof(scalar_t) == 4)
            {
                out << "\tscalarType\t32 bit;" << std::endl;
            }
            else if constexpr (sizeof(scalar_t) == 8)
            {
                out << "\tscalarType\t64 bit;" << std::endl;
            }
            out << "};" << std::endl;
            out << std::endl;

            // Write the field information: instantaneous or time-averaged, field names
            out << "fieldInformation" << std::endl;
            out << "{" << std::endl;
            out << "\ttimeStep\t" << timeStep << ";" << std::endl;
            out << std::endl;
            // For now, only writing instantaneous fields
            out << "\ttimeType\t" << timeTypeString<TimeType>() << ";" << std::endl;
            out << std::endl;
            out << "\tfieldNames[" << nVars << "]" << std::endl;
            out << "\t{" << std::endl;
            for (const auto &name : varNames)
            {
                out << "\t\t" << name << ";" << std::endl;
            }
            out << "\t};" << std::endl;
            out << "};" << std::endl;
            out << std::endl;

            // Write binary data with safe size conversion
            const std::size_t byteSize = expectedSize * sizeof(T);

            if (byteSize > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()))
            {
                throw std::runtime_error("Data size exceeds maximum stream size");
            }

            out << "fieldData" << std::endl;
            out << "{" << std::endl;
            out << "\tfieldType\tnonUniform;" << std::endl;
            out << std::endl;
            out << "\tfield[" << expectedSize << "][" << nVars << "][" << mesh.nz() << "][" << mesh.ny() << "][" << mesh.nx() << "]" << std::endl;
            out << "\t{" << std::endl;
            // out.flush();
            out.write(reinterpret_cast<const char *>(fields), static_cast<std::streamsize>(byteSize));
            out << std::endl;
            out << "\t};" << std::endl;
            out << "};" << std::endl;
        }
    }
}

#endif
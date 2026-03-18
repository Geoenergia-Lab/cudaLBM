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
        __host__ [[nodiscard]] const name_t timeTypeString() noexcept
        {
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
         * @param[in] fileName Name of the file to be written
         * @param[in] mesh The lattice mesh
         * @param[in] varNames The names of the solution variables
         * @param[in] fields The solution variables encoded in interleaved AoS format
         * @param[in] timeStep The current time step
         **/
        template <const time::type TimeType, class LatticeMesh, typename T>
        __host__ void writeFile(
            const name_t &fileName,
            const LatticeMesh &mesh,
            const words_t &varNames,
            const T *const ptrRestrict fields,
            const device::label_t timeStep,
            const device::label_t meanCount)
        {
            types::assertions::validate<T>();
            endian::assertions::validate();

            const host::label_t nVars = static_cast<host::label_t>(varNames.size());
            const host::label_t nPoints = mesh.template size<host::label_t>();
            const host::label_t expectedSize = nPoints * nVars;

            // Check if there is enough disk space to store the file
            fileSystem::diskSpaceAssertion<
                fileSystem::BINARY,
                fileSystem::fields::Yes,
                fileSystem::points::No,
                fileSystem::elements::No,
                fileSystem::offsets::No>(
                mesh,
                varNames.size(),
                fileName);

            std::ofstream out(fileName, std::ios::binary);
            if (!out)
            {
                throw std::runtime_error("Cannot open file: " + fileName);
            }

            // Write the system information: binary endianness
            out << "systemInformation" << std::endl;
            out << "{" << std::endl;
            out << "\tbinaryType\t" << endian::nameString() << ";" << std::endl;
            out << std::endl;
            out << "\tscalarSize\t" << sizeof(scalar_t) * 8 << ";" << std::endl;
            out << "};" << std::endl;
            out << std::endl;

            // Write the mesh information: number of points, number of devices
            static_assert(MULTI_GPU_ASSERTION(), MULTI_GPU_MSG_NOTE(fileIO::writeFile, "Multi-GPU must write GPU decomposition information to the file"));
            mesh.dimensions().print("latticeMesh", out);
            out << std::endl;
            mesh.nDevices().print("deviceDecomposition", out);
            out << std::endl;

            // Write the field information: instantaneous or time-averaged, field names
            out << "fieldInformation" << std::endl;
            out << "{" << std::endl;
            out << "\ttimeStep\t" << timeStep << ";" << std::endl;
            out << std::endl;
            // For now, only writing instantaneous fields
            out << "\ttimeType\t" << timeTypeString<TimeType>() << ";" << std::endl;
            out << std::endl;

            if constexpr (TimeType == time::timeAverage)
            {
                out << "\tmeanCount\t" << meanCount << ";" << std::endl;
                out << std::endl;
            }

            out << "\tnFields\t\t" << nVars << ";" << std::endl;
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
            const host::label_t byteSize = expectedSize * sizeof(T);

            if (byteSize > static_cast<host::label_t>(std::numeric_limits<std::streamsize>::max()))
            {
                throw std::runtime_error("Data size exceeds maximum stream size");
            }

            out << "fieldData" << std::endl;
            out << "{" << std::endl;
            out << "\tfieldType\tnonUniform;" << std::endl;
            out << std::endl;
            out << "\tfield[" << expectedSize << "][" << nVars << "][" << mesh.template dimension<axis::Z>() << "][" << mesh.template dimension<axis::Y>() << "][" << mesh.template dimension<axis::X>() << "]" << std::endl;
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
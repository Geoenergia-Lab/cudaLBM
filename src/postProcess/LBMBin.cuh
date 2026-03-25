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
    VTU binary file writer

Namespace
    LBM::postProcess

SourceFiles
    LBMBin.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_LBMBIN_CUH
#define __MBLBM_LBMBIN_CUH

namespace LBM
{
    namespace postProcess
    {
        class LBMBin : public writer
        {
        public:
            __host__ [[nodiscard]] static inline consteval fileSystem::format format() noexcept { return fileSystem::BINARY; }
            __host__ [[nodiscard]] static inline consteval fileSystem::fields::contained hasFields() noexcept { return fileSystem::fields::Yes; }
            __host__ [[nodiscard]] static inline consteval fileSystem::points::contained hasPoints() noexcept { return fileSystem::points::No; }
            __host__ [[nodiscard]] static inline consteval fileSystem::elements::contained hasElements() noexcept { return fileSystem::elements::No; }
            __host__ [[nodiscard]] static inline consteval fileSystem::offsets::contained hasOffsets() noexcept { return fileSystem::offsets::No; }
            __host__ [[nodiscard]] static inline consteval const char *fileExtension() noexcept { return ".LBMBin"; }
            __host__ [[nodiscard]] static inline consteval const char *name() noexcept { return "LBMBin"; }

            __host__ [[nodiscard]] inline consteval LBMBin(){};

            using This = LBMBin;

            /**
             * @brief Auxiliary template function that performs the VTU file writing.
             * @tparam indexType The data type for the mesh indices (uint32_t or host::label_t).
             **/
            template <const time::type TimeType, typename T>
            __host__ static void writeFile(
                const name_t &fileName,
                const host::latticeMesh &mesh,
                const words_t &varNames,
                const T *const ptrRestrict fields,
                const host::label_t timeStep,
                const host::label_t meanCount)
            {
                types::assertions::validate<T>();
                endian::assertions::validate();

                const host::label_t nVars = varNames.size();
                const host::label_t nPoints = mesh.size();
                const host::label_t expectedSize = nPoints * nVars;

                // Check if there is enough disk space to store the file
                writer::diskSpaceAssertion<This>(mesh, varNames, fileName);

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
                out << "\ttimeType\t" << fileIO::timeTypeString<TimeType>() << ";" << std::endl;
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
                out.write(reinterpret_cast<const char *>(fields), static_cast<std::streamsize>(byteSize));
                out << std::endl;
                out << "\t};" << std::endl;
                out << "};" << std::endl;
            }
        };
    }
}

#endif
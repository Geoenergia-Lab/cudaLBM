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
            template <typename T>
            __host__ static void write(
                const name_t &fileName,
                const host::latticeMesh &mesh,
                const words_t &varNames,
                const T *const ptrRestrict fields,
                const host::label_t timeStep,
                const host::label_t meanCount)
            {
                common_write<time::timeAverage>(fileName, mesh, varNames, fields, timeStep, meanCount);
            }

            template <typename T>
            __host__ static void write(
                const name_t &fileName,
                const host::latticeMesh &mesh,
                const words_t &varNames,
                const T *const ptrRestrict fields,
                const host::label_t timeStep)
            {
                common_write<time::instantaneous>(fileName, mesh, varNames, fields, timeStep, 0);
            }

        private:
            template <const time::type TimeType, typename T>
            __host__ static void common_write(
                const name_t &fileName,
                const host::latticeMesh &mesh,
                const words_t &varNames,
                const T *const ptrRestrict fields,
                const host::label_t timeStep,
                const host::label_t meanCount)
            {
                types::assertions::validate<T>();
                endian::assertions::validate();

                // Check if there is enough disk space to store the file
                writer::diskSpaceAssertion<This>(mesh, varNames, fileName);

                std::ofstream out(fileName, std::ios::binary);
                if (!out)
                {
                    throw std::runtime_error("Cannot open file: " + fileName);
                }

                // Write the system information
                systemInfo::print(out);

                // Write the mesh information: number of points, number of devices
                mesh.dimensions().print<true>("latticeMesh", out);
                mesh.nDevices().print<true>("deviceDecomposition", out);

                // Write the field information: instantaneous or time-averaged, field names
                writeFieldInformation<TimeType>(timeStep, varNames, meanCount, out);

                // Write binary data with safe size conversion
                writeFieldData(mesh, fields, varNames, out);
            }

            template <typename T>
            __host__ static void writeFieldData(
                const host::latticeMesh &mesh,
                const T *const ptrRestrict fields,
                const words_t &varNames,
                std::ofstream &out)
            {
                const host::label_t nPoints = mesh.size();
                const host::label_t expectedSize = nPoints * varNames.size();
                const host::label_t byteSize = expectedSize * sizeof(T);

                out << "fieldData" << std::endl;
                out << "{" << std::endl;
                out << "    fieldType\tnonUniform;" << std::endl;
                out << std::endl;
                out << "    field[" << expectedSize << "][" << varNames.size() << "][" << mesh.template dimension<axis::Z>() << "][" << mesh.template dimension<axis::Y>() << "][" << mesh.template dimension<axis::X>() << "]" << std::endl;
                out << "    {" << std::endl;

                fileIO::writeBinaryBlock(fields, byteSize, out);

                out << std::endl;
                out << "    };" << std::endl;
                out << "};" << std::endl;
            }

            template <const time::type TimeType>
            __host__ static void writeFieldInformation(
                const host::label_t timeStep,
                const words_t &varNames,
                const host::label_t meanCount,
                std::ofstream &out)
            {
                out << "fieldInformation" << std::endl;
                out << "{" << std::endl;
                out << "    timeStep\t" << timeStep << ";" << std::endl;
                out << std::endl;
                // For now, only writing instantaneous fields
                out << "    timeType\t" << fileIO::timeTypeString<TimeType>() << ";" << std::endl;
                out << std::endl;

                if constexpr (TimeType == time::timeAverage)
                {
                    out << "    meanCount\t" << meanCount << ";" << std::endl;
                    out << std::endl;
                }

                out << "    nFields\t\t" << varNames.size() << ";" << std::endl;
                out << std::endl;
                out << "    fieldNames[" << varNames.size() << "]" << std::endl;
                out << "    {" << std::endl;
                for (const auto &name : varNames)
                {
                    out << "    \t" << name << ";" << std::endl;
                }
                out << "    };" << std::endl;
                out << "};" << std::endl;
                out << std::endl;
            }
        };
    }
}

#endif
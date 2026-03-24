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
    VTU.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_VTU_CUH
#define __MBLBM_VTU_CUH

namespace LBM
{
    namespace postProcess
    {
        class VTU : public writer
        {
        public:
            __host__ [[nodiscard]] static inline consteval fileSystem::format format() noexcept { return fileSystem::BINARY; }
            __host__ [[nodiscard]] static inline consteval fileSystem::fields::contained hasFields() noexcept { return fileSystem::fields::Yes; }
            __host__ [[nodiscard]] static inline consteval fileSystem::points::contained hasPoints() noexcept { return fileSystem::points::Yes; }
            __host__ [[nodiscard]] static inline consteval fileSystem::elements::contained hasElements() noexcept { return fileSystem::elements::Yes; }
            __host__ [[nodiscard]] static inline consteval fileSystem::offsets::contained hasOffsets() noexcept { return fileSystem::offsets::Yes; }
            __host__ [[nodiscard]] static inline consteval const char *fileExtension() noexcept { return ".vtu"; }
            __host__ [[nodiscard]] static inline consteval const char *name() noexcept { return "VTU"; }

            __host__ [[nodiscard]] inline consteval VTU(){};

            // Write implementation
            __host__ static bool write(
                const std::vector<std::vector<scalar_t>> &solutionVars,
                std::ofstream &outFile,
                const host::latticeMesh &mesh,
                const words_t &varNames) noexcept
            {
                const host::label_t numNodes = mesh.dimension<axis::X>() * mesh.dimension<axis::Y>() * mesh.dimension<axis::Z>();
                const host::label_t numElements = (mesh.dimension<axis::X>() - 1) * (mesh.dimension<axis::Y>() - 1) * (mesh.dimension<axis::Z>() - 1);
                const host::label_t numVars = solutionVars.size();

                std::cout << "Creating mesh detail" << std::endl;
                const std::vector<scalar_t> points = meshCoordinates<scalar_t>(mesh);
                const std::vector<host::label_t> connectivity = meshConnectivity<false, host::label_t>(mesh);
                const std::vector<host::label_t> offsets = meshOffsets<host::label_t>(mesh);
                std::cout << "Done creating mesh detail" << std::endl;
                std::cout << "points.size() = " << points.size() << std::endl;
                std::cout << "connectivity.size() = " << connectivity.size() << std::endl;
                std::cout << "offsets.size() = " << offsets.size() << std::endl;

                std::stringstream xml;
                host::label_t currentOffset = 0;

                xml << "<?xml version=\"1.0\"?>\n";
                xml << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n";
                xml << "  <UnstructuredGrid>\n";
                xml << "    <Piece NumberOfPoints=\"" << numNodes << "\" NumberOfCells=\"" << numElements << "\">\n";

                xml << "      <PointData Scalars=\"" << (varNames.empty() ? "" : varNames[0]) << "\">\n";
                for (host::label_t i = 0; i < numVars; ++i)
                {
                    xml << "        <DataArray type=\"" << getVtkTypeName<scalar_t>() << "\" Name=\"" << varNames[i] << "\" format=\"appended\" offset=\"" << currentOffset << "\"/>\n";
                    currentOffset += sizeof(host::label_t) + solutionVars[i].size() * sizeof(scalar_t);
                }
                xml << "      </PointData>\n";

                xml << "      <Points>\n";
                xml << "        <DataArray type=\"" << getVtkTypeName<scalar_t>() << "\" Name=\"Coordinates\" NumberOfComponents=\"3\" format=\"appended\" offset=\"" << currentOffset << "\"/>\n";
                xml << "      </Points>\n";
                currentOffset += sizeof(host::label_t) + points.size() * sizeof(scalar_t);

                xml << "      <Cells>\n";
                // Usa o indexType para obter o nome do tipo VTK correto
                xml << "        <DataArray type=\"" << getVtkTypeName<host::label_t>() << "\" Name=\"connectivity\" format=\"appended\" offset=\"" << currentOffset << "\"/>\n";
                currentOffset += sizeof(host::label_t) + connectivity.size() * sizeof(host::label_t);

                xml << "        <DataArray type=\"" << getVtkTypeName<host::label_t>() << "\" Name=\"offsets\" format=\"appended\" offset=\"" << currentOffset << "\"/>\n";
                currentOffset += sizeof(host::label_t) + offsets.size() * sizeof(host::label_t);

                xml << "        <DataArray type=\"" << getVtkTypeName<uint8_t>() << "\" Name=\"types\" format=\"appended\" offset=\"" << currentOffset << "\"/>\n";
                xml << "      </Cells>\n";

                xml << "    </Piece>\n";
                xml << "  </UnstructuredGrid>\n";
                xml << "  <AppendedData encoding=\"raw\">_";

                outFile << xml.str();

                for (host::label_t i = 0; i < solutionVars.size(); i++)
                {
                    std::cout << varNames[i] << std::endl;
                    writeBinaryBlock(solutionVars[i], outFile);
                }

                // for (const auto &varData : solutionVars)
                // {
                //     writeBinaryBlock(varData, outFile);
                // }
                std::cout << "points" << std::endl;
                writeBinaryBlock(points, outFile);

                std::cout << "connectivity" << std::endl;
                writeBinaryBlock(connectivity, outFile);

                std::cout << "offsets" << std::endl;
                writeBinaryBlock(offsets, outFile);

                const std::vector<uint8_t> types(numElements, 12); // 12 é o código VTK para hexaedro

                std::cout << "types" << std::endl;
                writeBinaryBlock(types, outFile);

                outFile << "</AppendedData>\n";
                outFile << "</VTKFile>\n";

                outFile.close();

                return outFile.good();
            }
        };
    }
}

#endif
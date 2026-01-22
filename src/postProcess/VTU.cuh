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
        namespace VTU
        {
            __host__ [[nodiscard]] inline consteval bool hasFields() { return true; }
            __host__ [[nodiscard]] inline consteval bool hasPoints() { return true; }
            __host__ [[nodiscard]] inline consteval bool hasElements() { return true; }
            __host__ [[nodiscard]] inline consteval bool hasOffsets() { return true; }
            __host__ [[nodiscard]] inline consteval const char *fileExtension() { return ".vtu"; }

            /**
             * @brief Auxiliary template function that performs the VTU file writing.
             * @tparam indexType The data type for the mesh indices (uint32_t or uint64_t).
             */
            template <typename indexType>
            __host__ void VTUWriter(
                const std::vector<std::vector<scalar_t>> &solutionVars,
                std::ofstream &outFile,
                const host::latticeMesh &mesh,
                const std::vector<std::string> &solutionVarNames) noexcept
            {
                const label_t numNodes = mesh.nx() * mesh.ny() * mesh.nz();
                const label_t numElements = (mesh.nx() - 1) * (mesh.ny() - 1) * (mesh.nz() - 1);
                const std::size_t numVars = solutionVars.size();

                const std::vector<scalar_t> points = meshCoordinates<scalar_t>(mesh);
                const std::vector<indexType> connectivity = meshConnectivity<false, indexType>(mesh);
                const std::vector<indexType> offsets = meshOffsets<indexType>(mesh);

                std::stringstream xml;
                uint64_t currentOffset = 0;

                xml << "<?xml version=\"1.0\"?>\n";
                xml << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n";
                xml << "  <UnstructuredGrid>\n";
                xml << "    <Piece NumberOfPoints=\"" << numNodes << "\" NumberOfCells=\"" << numElements << "\">\n";

                xml << "      <PointData Scalars=\"" << (solutionVarNames.empty() ? "" : solutionVarNames[0]) << "\">\n";
                for (std::size_t i = 0; i < numVars; ++i)
                {
                    xml << "        <DataArray type=\"" << getVtkTypeName<scalar_t>() << "\" Name=\"" << solutionVarNames[i] << "\" format=\"appended\" offset=\"" << currentOffset << "\"/>\n";
                    currentOffset += sizeof(uint64_t) + solutionVars[i].size() * sizeof(scalar_t);
                }
                xml << "      </PointData>\n";

                xml << "      <Points>\n";
                xml << "        <DataArray type=\"" << getVtkTypeName<scalar_t>() << "\" Name=\"Coordinates\" NumberOfComponents=\"3\" format=\"appended\" offset=\"" << currentOffset << "\"/>\n";
                xml << "      </Points>\n";
                currentOffset += sizeof(uint64_t) + points.size() * sizeof(scalar_t);

                xml << "      <Cells>\n";
                // Usa o indexType para obter o nome do tipo VTK correto
                xml << "        <DataArray type=\"" << getVtkTypeName<indexType>() << "\" Name=\"connectivity\" format=\"appended\" offset=\"" << currentOffset << "\"/>\n";
                currentOffset += sizeof(uint64_t) + connectivity.size() * sizeof(indexType);

                xml << "        <DataArray type=\"" << getVtkTypeName<indexType>() << "\" Name=\"offsets\" format=\"appended\" offset=\"" << currentOffset << "\"/>\n";
                currentOffset += sizeof(uint64_t) + offsets.size() * sizeof(indexType);

                xml << "        <DataArray type=\"" << getVtkTypeName<uint8_t>() << "\" Name=\"types\" format=\"appended\" offset=\"" << currentOffset << "\"/>\n";
                xml << "      </Cells>\n";

                xml << "    </Piece>\n";
                xml << "  </UnstructuredGrid>\n";
                xml << "  <AppendedData encoding=\"raw\">_";

                outFile << xml.str();

                for (const auto &varData : solutionVars)
                {
                    writeBinaryBlock(varData, outFile);
                }
                writeBinaryBlock(points, outFile);
                writeBinaryBlock(connectivity, outFile);
                writeBinaryBlock(offsets, outFile);

                const std::vector<uint8_t> types(numElements, 12); // 12 é o código VTK para hexaedro
                writeBinaryBlock(types, outFile);

                outFile << "</AppendedData>\n";
                outFile << "</VTKFile>\n";

                outFile.close();
            }

            /**
             * @brief Writes solution variables to an unstructured grid VTU file (.vtu)
             * This function checks the mesh size and dispatches to the implementation with
             * the appropriate index type (32-bit or 64-bit).
             */
            __host__ void write(
                const std::vector<std::vector<scalar_t>> &solutionVars,
                const std::string &fileName,
                const host::latticeMesh &mesh,
                const std::vector<std::string> &solutionVarNames)
            {
                const uint64_t numNodes = static_cast<uint64_t>(mesh.nx()) * static_cast<uint64_t>(mesh.ny()) * static_cast<uint64_t>(mesh.nz());
                const std::size_t numVars = solutionVars.size();

                if (numVars != solutionVarNames.size())
                {
                    errorHandler(-1, "Error: The number of solution (" + std::to_string(numVars) + ") does not match the count of variable names (" + std::to_string(solutionVarNames.size()));
                }

                for (std::size_t i = 0; i < numVars; i++)
                {
                    if (solutionVars[i].size() != numNodes)
                    {
                        errorHandler(-1, "Error: The solution variable " + std::to_string(i) + " has " + std::to_string(solutionVars[i].size()) + " elements, expected " + std::to_string(numNodes));
                    }
                }

                std::cout << "vtuWriter:" << std::endl;
                std::cout << "{" << std::endl;
                std::cout << "    fileName: " << directoryPrefix() << "/" << fileName << fileExtension() << ";" << std::endl;

                if (!std::filesystem::is_directory(directoryPrefix()))
                {
                    if (!std::filesystem::create_directory(directoryPrefix()))
                    {
                        std::cout << "    directoryStatus: Unable to create directory" << directoryPrefix() << ";" << std::endl;
                        std::cout << "    writeStatus: Fail (unable to create directory)" << ";" << std::endl;
                        std::cout << "};" << std::endl;
                        errorHandler(-1, "Error: Unable to create directory" + std::string(directoryPrefix()));
                    }
                }
                else
                {
                    std::cout << "    directoryStatus: OK;" << std::endl;
                }

                std::cout << "    fileSize: " << fileSystem::to_mebibytes<double>(fileSystem::expectedDiskUsage<fileSystem::BINARY, hasFields(), hasPoints(), hasElements(), hasOffsets()>(mesh, solutionVars.size())) << " MiB;" << std::endl;

                // Check if there is enough disk space to store the file
                if (!fileSystem::diskSpaceCheck<fileSystem::ASCII, hasFields(), hasPoints(), hasElements(), hasOffsets()>(mesh, solutionVars.size()))
                {
                    std::cout << "    diskSpace: Insufficient (" << fileSystem::to_mebibytes<double>(fileSystem::availableDiskSpace()) << " MiB);" << std::endl;
                    std::cout << "    writeStatus: Fail (insufficient disk space)" << ";" << std::endl;
                    std::cout << "};" << std::endl;
                    errorHandler(-1, "Error: Insufficient disk space on drive " + fileSystem::diskName());
                }
                else
                {
                    std::cout << "    diskSpace: OK (" << fileSystem::to_mebibytes<double>(fileSystem::availableDiskSpace()) << " MiB);" << std::endl;
                }

                // Check if there is enough disk space to store the file
                fileSystem::diskSpaceAssertion<fileSystem::BINARY, hasFields(), hasPoints(), hasElements(), hasOffsets()>(mesh, solutionVars.size(), fileName);

                constexpr const uint64_t limit32 = static_cast<uint64_t>(std::numeric_limits<uint32_t>::max());

                std::cout << "    indexType: " << ((numNodes >= limit32) ? "uint64_t;" : "uint32_t;") << std::endl;

                const std::string trueFileName(std::string(directoryPrefix()) + "/" + fileName + fileExtension());

                std::ofstream outFile(trueFileName);
                if (outFile)
                {
                    std::cout << "    ofstreamStatus: OK;" << std::endl;
                }
                else
                {
                    std::cout << "    ofstreamStatus: Fail" << std::endl;
                    std::cout << "};" << std::endl;
                    errorHandler(-1, "Error opening file: " + trueFileName);
                }

                if (numNodes >= limit32)
                {
                    VTUWriter<uint64_t>(solutionVars, outFile, mesh, solutionVarNames);
                }
                else
                {
                    VTUWriter<uint32_t>(solutionVars, outFile, mesh, solutionVarNames);
                }
                std::cout << "    writeStatus: success" << ";" << std::endl;
                std::cout << "};" << std::endl;
                std::cout << std::endl;
            }
        }
    }
}

#endif
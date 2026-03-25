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
    VTS binary file writer

Namespace
    LBM::postProcess

SourceFiles
    VTS.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_VTS_CUH
#define __MBLBM_VTS_CUH

namespace LBM
{
    namespace postProcess
    {
        class VTS : public writer
        {
        public:
            __host__ [[nodiscard]] static inline consteval fileSystem::format format() noexcept { return fileSystem::BINARY; }
            __host__ [[nodiscard]] static inline consteval fileSystem::fields::contained hasFields() noexcept { return fileSystem::fields::Yes; }
            __host__ [[nodiscard]] static inline consteval fileSystem::points::contained hasPoints() noexcept { return fileSystem::points::Yes; }
            __host__ [[nodiscard]] static inline consteval fileSystem::elements::contained hasElements() noexcept { return fileSystem::elements::No; }
            __host__ [[nodiscard]] static inline consteval fileSystem::offsets::contained hasOffsets() noexcept { return fileSystem::offsets::No; }
            __host__ [[nodiscard]] static inline consteval const char *fileExtension() noexcept { return ".vts"; }
            __host__ [[nodiscard]] static inline consteval const char *name() noexcept { return "VTS"; }

            __host__ [[nodiscard]] inline consteval VTS(){};

            /**
             * @brief Auxiliary template function that performs the VTU file writing.
             **/
            __host__ static bool write(
                const std::vector<std::vector<scalar_t>> &solutionVars,
                std::ofstream &outFile,
                const host::latticeMesh &mesh,
                const words_t &varNames)
            {
                // For a structured grid, we need different calculations
                const host::label_t numVars = solutionVars.size();

                // Get points in the correct order for structured grid (i fastest, then j, then k)
                const std::vector<scalar_t> points = meshCoordinates<scalar_t>(mesh);

                {
                    std::stringstream xml;
                    host::label_t currentOffset = 0;

                    // Calculate extents - note the -1 for the maximum indices
                    const host::label_t dimX = mesh.dimension<axis::X>() - 1;
                    const host::label_t dimY = mesh.dimension<axis::Y>() - 1;
                    const host::label_t dimZ = mesh.dimension<axis::Z>() - 1;

                    xml << "<?xml version=\"1.0\"?>\n";
                    xml << "<VTKFile type=\"StructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n";
                    xml << "  <StructuredGrid WholeExtent=\"0 " << dimX << " 0 " << dimY << " 0 " << dimZ << "\">\n";
                    xml << "    <Piece Extent=\"0 " << dimX << " 0 " << dimY << " 0 " << dimZ << "\">\n";

                    // Point data (same as before)
                    xml << "      <PointData Scalars=\"" << (varNames.empty() ? "" : varNames[0]) << "\">\n";
                    for (host::label_t i = 0; i < numVars; ++i)
                    {
                        xml << "        <DataArray type=\"" << getVtkTypeName<scalar_t>() << "\" Name=\"" << varNames[i] << "\" format=\"appended\" offset=\"" << currentOffset << "\"/>\n";
                        currentOffset += sizeof(host::label_t) + solutionVars[i].size() * sizeof(scalar_t);
                    }
                    xml << "      </PointData>\n";

                    // Points section (same as before)
                    xml << "      <Points>\n";
                    xml << "        <DataArray type=\"" << getVtkTypeName<scalar_t>() << "\" Name=\"Coordinates\" NumberOfComponents=\"" << 3 << "\" format=\"appended\" offset=\"" << currentOffset << "\"/>\n";
                    xml << "      </Points>\n";
                    currentOffset += sizeof(host::label_t) + points.size() * sizeof(scalar_t);

                    xml << "    </Piece>\n";
                    xml << "  </StructuredGrid>\n";
                    xml << "  <AppendedData encoding=\"raw\">_";

                    outFile << xml.str();
                }

                // Write point data arrays
                for (const auto &varData : solutionVars)
                {
                    fileIO::writeBinaryBlock(varData, outFile);
                }

                // Write points
                fileIO::writeBinaryBlock(points, outFile);

                outFile << "</AppendedData>\n";
                outFile << "</VTKFile>\n";

                outFile.close();

                return outFile.good();
            }
        };
    }
}

#endif
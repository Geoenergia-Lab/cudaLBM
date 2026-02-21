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
    Tecplot ASCII file writer

Namespace
    LBM::postProcess

SourceFiles
    Tecplot.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_TECPLOT_CUH
#define __MBLBM_TECPLOT_CUH

namespace LBM
{
    namespace postProcess
    {
        namespace Tecplot
        {
            __host__ [[nodiscard]] inline consteval fileSystem::format format() noexcept { return fileSystem::ASCII; }
            __host__ [[nodiscard]] inline consteval fileSystem::fields::contained hasFields() noexcept { return fileSystem::fields::Yes; }
            __host__ [[nodiscard]] inline consteval fileSystem::points::contained hasPoints() noexcept { return fileSystem::points::Yes; }
            __host__ [[nodiscard]] inline consteval fileSystem::elements::contained hasElements() noexcept { return fileSystem::elements::Yes; }
            __host__ [[nodiscard]] inline consteval fileSystem::offsets::contained hasOffsets() noexcept { return fileSystem::offsets::Yes; }
            __host__ [[nodiscard]] inline consteval const char *fileExtension() noexcept { return ".dat"; }

            /**
             * @brief Writes solution data to a Tecplot ASCII file in unstructured grid format
             * @param[in] solutionVars Vector of solution variable arrays (Structure of Arrays format)
             * @param[in] fileName Output filename for Tecplot data
             * @param[in] mesh The lattice mesh
             * @param[in] solutionVarNames Names of the solution variables for Tecplot header
             * @param[in] title Title for the Tecplot file
             * @return None
             * @note Uses 1-based indexing for element connectivity (Tecplot convention)
             * @note Output format: BLOCK data packing with FEBRICK (hexahedral) elements
             * @note Uses high precision (50 digits) for numerical output
             *
             * This function writes simulation results to a Tecplot-compatible ASCII file
             * with the following structure:
             * 1. File header with title and variable declarations
             * 2. Coordinate data in separate blocks (X, Y, Z)
             * 3. Solution variables in separate blocks
             * 4. Element connectivity with 1-based indexing
             *
             * The function performs comprehensive validation of input data including:
             * - Variable count matching name count
             * - Node count consistency across all arrays
             * - File accessibility checks
             **/
            __host__ void writeTecplot(
                const std::vector<std::vector<scalar_t>> &solutionVars,
                std::ofstream &outFile,
                const host::latticeMesh &mesh,
                const words_t &solutionVarNames) noexcept
            {
                // Check input sizes
                const label_t numNodes = mesh.dimension<axis::X>() * mesh.dimension<axis::Y>() * mesh.dimension<axis::Z>();

                // Set high precision output
                outFile << std::setprecision(std::numeric_limits<scalar_t>::max_digits10);

                // Write Tecplot header
                // outFile << "TITLE = \"" << title << "\"\n";
                outFile << "VARIABLES = \"X\" \"Y\" \"Z\" ";
                for (auto &name : solutionVarNames)
                {
                    outFile << "\"" << name << "\" ";
                }
                outFile << "\n";

                // UNSTRUCTURED GRID FORMAT
                const label_t numElements = (mesh.dimension<axis::X>() - 1) * (mesh.dimension<axis::Y>() - 1) * (mesh.dimension<axis::Z>() - 1);
                outFile << "ZONE T=\"Hexahedral Zone\", NODES=" << numNodes << ", ELEMENTS=" << numElements << ", DATAPACKING=BLOCK, ZONETYPE=FEBRICK\n";

                const std::vector<scalar_t> coords = meshCoordinates<scalar_t>(mesh);

                // Write node coordinates (X, Y, Z blocks)
                // Write X
                for (label_t n = 0; n < numNodes; ++n)
                {
                    outFile << coords[3 * n + 0] << "\n";
                }

                // Write Y
                for (label_t n = 0; n < numNodes; ++n)
                {
                    outFile << coords[3 * n + 1] << "\n";
                }

                // Write Z
                for (label_t n = 0; n < numNodes; ++n)
                {
                    outFile << coords[3 * n + 2] << "\n";
                }

                // Write solution variables (each as a separate block)
                for (const auto &varData : solutionVars)
                {
                    for (const auto &value : varData)
                    {
                        outFile << value << "\n";
                    }
                }

                const std::vector<label_t> connectivity = meshConnectivity<true, label_t>(mesh);
                for (label_t e = 0; e < numElements; ++e)
                {
                    for (label_t n = 0; n < 8; ++n)
                    {
                        outFile << connectivity[e * 8 + n] << (n < 7 ? " " : "\n");
                    }
                }

                outFile.close();
            }

            __host__ void write(
                const std::vector<std::vector<scalar_t>> &solutionVars,
                const name_t &fileName,
                const host::latticeMesh &mesh,
                const words_t &solutionVarNames)
            {
                const uint64_t numNodes = mesh.dimension<axis::X, uint64_t>() * mesh.dimension<axis::Y, uint64_t>() * mesh.dimension<axis::Z, uint64_t>();
                const std::size_t numVars = solutionVars.size();

                if (numVars != solutionVarNames.size())
                {
                    throw std::runtime_error("Error: The number of solution (" + std::to_string(numVars) + ") does not match the count of variable names (" + std::to_string(solutionVarNames.size()));
                }

                for (std::size_t i = 0; i < numVars; i++)
                {
                    if (solutionVars[i].size() != numNodes)
                    {
                        throw std::runtime_error("Error: The solution variable " + std::to_string(i) + " has " + std::to_string(solutionVars[i].size()) + " elements, expected " + std::to_string(numNodes));
                    }
                }

                std::cout << "TecplotWriter:" << std::endl;
                std::cout << "{" << std::endl;
                std::cout << "    fileName: " << directoryPrefix() << "/" << fileName << fileExtension() << ";" << std::endl;

                if (!std::filesystem::is_directory(directoryPrefix()))
                {
                    if (!std::filesystem::create_directory(directoryPrefix()))
                    {
                        std::cout << "    directoryStatus: unable to create directory" << directoryPrefix() << ";" << std::endl;
                        std::cout << "    writeStatus: fail (unable to create directory)" << ";" << std::endl;
                        std::cout << "};" << std::endl;
                        throw std::runtime_error("Error: unable to create directory" + name_t(directoryPrefix()));
                    }
                }
                else
                {
                    std::cout << "    directoryStatus: OK;" << std::endl;
                }

                std::cout << "    fileSize: "
                          << fileSystem::to_mebibytes<double>(
                                 fileSystem::expectedDiskUsage<
                                     format(),
                                     hasFields(),
                                     hasPoints(),
                                     hasElements(),
                                     hasOffsets()>(
                                     mesh,
                                     solutionVars.size()))
                          << " MiB;" << std::endl;

                // Check if there is enough disk space to store the file
                fileSystem::diskSpaceAssertion<
                    format(),
                    hasFields(),
                    hasPoints(),
                    hasElements(),
                    hasOffsets()>(
                    mesh,
                    solutionVars.size(),
                    fileName);

                const name_t trueFileName(name_t(directoryPrefix()) + "/" + fileName + fileExtension());

                std::ofstream outFile(trueFileName);
                if (outFile)
                {
                    std::cout << "    ofstreamStatus: OK;" << std::endl;
                }
                else
                {
                    std::cout << "    ofstreamStatus: Fail" << std::endl;
                    std::cout << "};" << std::endl;
                    throw std::runtime_error("Error opening file: " + trueFileName);
                }

                writeTecplot(solutionVars, outFile, mesh, solutionVarNames);

                std::cout << "    writeStatus: success" << ";" << std::endl;
                std::cout << "};" << std::endl;
                std::cout << std::endl;
            }
        }
    }
}

#endif
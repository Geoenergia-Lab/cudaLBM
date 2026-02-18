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
    Post-processing utility to calculate derived fields from saved moment fields
    Supported calculations: velocity magnitude, velocity divergence, vorticity,
    vorticity magnitude, integrated vorticity

Namespace
    LBM

SourceFiles
    testExecutable.cu

\*---------------------------------------------------------------------------*/

#include "testExecutable.cuh"

using namespace LBM;

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

int main()
{
    // constexpr const label_t q = 26;

    // std::cout << VelocitySet::cx<int>(q_i<q>()) << std::endl;

    const name_t fileName = "jetFlow_20000.LBMBin";

    const words_t lines = read_first_n_lines(fileName, 50);

    // const fileIO::systemInformation sysInfo(string::extractBlock(lines, "systemInformation", 0));

    // std::cout << sysInfo.endianType() << std::endl;
    // std::cout << sysInfo.scalarSize() << std::endl;

    // const fileIO::meshPrimitive mesh(string::extractBlock(lines, "latticeMesh", 0));

    // mesh.nPoints().print("nPoints");
    // mesh.nDevices().print("nDevices");

    const fileIO::fieldInformation fieldInfo(string::extractBlock(lines, "fieldInformation", 0));

    std::cout << fieldInfo.meanCount() << std::endl;

    // std::cout << "timeStep: " << fieldInfo.timeStep() << std::endl;
    // std::cout << "timeType: " << (fieldInfo.timeType() == time::instantaneous ? "instantaneous" : "timeAverage") << std::endl;
    // std::cout << "nFields: " << fieldInfo.nFields() << std::endl;

    // for (std::size_t i = 0; i < fieldInfo.fieldNames().size(); i++)
    // {
    //     std::cout << fieldInfo.fieldNames()[i] << std::endl;
    // }

    return 0;
}
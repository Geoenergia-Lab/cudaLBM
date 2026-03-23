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
    fieldCalculate.cu

\*---------------------------------------------------------------------------*/

#include "fieldCalculate.cuh"

using namespace LBM;

int main(const int argc, const char *const argv[])
{
    const programControl programCtrl(argc, argv);

    const host::latticeMesh mesh(programCtrl);

    // Check if calculation type argument is present
    const bool calculationType = programCtrl.input().isArgPresent("-calculationType");

    // Parse the argument if present, otherwise set to empty string
    const name_t calculationTypeString = calculationType ? programCtrl.getArgument("-calculationType") : "";

    if (calculationTypeString == "containsNaN")
    {
        // Get the time indices
        const std::vector<host::label_t> fileNameIndices = fileIO::timeIndices(programCtrl.caseName());

        for (host::label_t timeStep = fileIO::getStartIndex(programCtrl.caseName(), programCtrl); timeStep < fileNameIndices.size(); timeStep++)
        {
            // We should check for field names here. Currently we are just doing the default fields
            const host::arrayCollection<scalar_t, ctorType::MUST_READ> hostMoments(
                programCtrl,
                {"rho", "u", "v", "w", "m_xx", "m_xy", "m_xz", "m_yy", "m_yz", "m_zz"},
                timeStep);

            containsNaN(hostMoments, mesh, fileNameIndices[timeStep]);

            if (timeStep < fileNameIndices.size() - 1)
            {
                std::cout << std::endl;
            }
        }
    }

    if (calculationTypeString == "spatialMean")
    {
        // Get the time indices
        const std::vector<host::label_t> fileNameIndices = fileIO::timeIndices(programCtrl.caseName());

        for (host::label_t timeStep = fileIO::getStartIndex(programCtrl.caseName(), programCtrl); timeStep < fileNameIndices.size(); timeStep++)
        {
            // We should check for field names here. Currently we are just doing the default fields
            const host::arrayCollection<scalar_t, ctorType::MUST_READ> hostMoments(
                programCtrl,
                {"rho", "u", "v", "w", "m_xx", "m_xy", "m_xz", "m_yy", "m_yz", "m_zz"},
                timeStep);

            spatialMean(hostMoments, mesh, fileNameIndices[timeStep]);

            if (timeStep < fileNameIndices.size() - 1)
            {
                std::cout << std::endl;
            }
        }
    }

    if (calculationTypeString == "vorticity")
    {
        // Get the conversion type
        const name_t conversion = programCtrl.getArgument("-fileType");

        // Get the writer function
        const std::unordered_map<name_t, postProcess::writerFunction>::const_iterator it = postProcess::writers.find(conversion);

        // Get the time indices
        const std::vector<host::label_t> fileNameIndices = fileIO::timeIndices(programCtrl.caseName());

        if (it != postProcess::writers.end())
        {
            for (host::label_t timeStep = fileIO::getStartIndex(programCtrl.caseName(), programCtrl); timeStep < fileNameIndices.size(); timeStep++)
            {
                // Get the file name at the present time step
                const name_t fileName = "vorticity_" + std::to_string(fileNameIndices[timeStep]);

                const host::arrayCollection<scalar_t, ctorType::MUST_READ> hostMoments(
                    programCtrl,
                    {"rho", "u", "v", "w", "m_xx", "m_xy", "m_xz", "m_yy", "m_yz", "m_zz"},
                    timeStep);

                const std::vector<std::vector<scalar_t>> fields = fileIO::deinterleaveAoS(hostMoments.arr(), mesh);

                const std::vector<std::vector<scalar_t>> omega = numericalSchemes::derivative::curl<SchemeOrder()>(fields[index::u], fields[index::v], fields[index::w], mesh);
                const std::vector<scalar_t> magomega = numericalSchemes::mag(omega[0], omega[1], omega[2]);

                const postProcess::writerFunction writer = it->second;

                writer({omega[0], omega[1], omega[2], magomega}, fileName, mesh, {"omega_x", "omega_y", "omega_z", "mag[omega]"});

                if (timeStep < fileNameIndices.size() - 1)
                {
                    std::cout << std::endl;
                }
            }
        }
    }

    if (calculationTypeString == "div[U]")
    {
        // Get the conversion type
        const name_t conversion = programCtrl.getArgument("-fileType");

        // Get the writer function
        const std::unordered_map<name_t, postProcess::writerFunction>::const_iterator it = postProcess::writers.find(conversion);

        // Get the time indices
        const std::vector<host::label_t> fileNameIndices = fileIO::timeIndices(programCtrl.caseName());

        if (it != postProcess::writers.end())
        {
            for (host::label_t timeStep = fileIO::getStartIndex(programCtrl.caseName(), programCtrl); timeStep < fileNameIndices.size(); timeStep++)
            {
                // Get the file name at the present time step
                const name_t fileName = "div[U]_" + std::to_string(fileNameIndices[timeStep]);

                const host::arrayCollection<scalar_t, ctorType::MUST_READ> hostMoments(
                    programCtrl,
                    {"rho", "u", "v", "w", "m_xx", "m_xy", "m_xz", "m_yy", "m_yz", "m_zz"},
                    timeStep);

                const std::vector<std::vector<scalar_t>> fields = fileIO::deinterleaveAoS(hostMoments.arr(), mesh);

                const std::vector<scalar_t> divu = numericalSchemes::derivative::div<SchemeOrder()>(fields[index::u], fields[index::v], fields[index::w], mesh);

                const postProcess::writerFunction writer = it->second;

                writer({divu}, fileName, mesh, {"div[U]"});

                if (timeStep < fileNameIndices.size() - 1)
                {
                    std::cout << std::endl;
                }
            }
        }
    }

    return 0;
}
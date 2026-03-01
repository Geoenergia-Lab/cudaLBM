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
        const std::vector<label_t> fileNameIndices = fileIO::timeIndices(programCtrl.caseName());

        for (label_t timeStep = fileIO::getStartIndex(programCtrl.caseName(), programCtrl); timeStep < fileNameIndices.size(); timeStep++)
        {
            // We should check for field names here. Currently we are just doing the default fields
            const host::arrayCollection<scalar_t, ctorType::MUST_READ> hostMoments(
                programCtrl,
                {"rho", "u", "v", "w", "m_xx", "m_xy", "m_xz", "m_yy", "m_yz", "m_zz"},
                timeStep);

            containsNaN(hostMoments, mesh, fileNameIndices[timeStep]);
        }
    }

    if (calculationTypeString == "spatialMean")
    {
        // Get the time indices
        const std::vector<label_t> fileNameIndices = fileIO::timeIndices(programCtrl.caseName());

        for (label_t timeStep = fileIO::getStartIndex(programCtrl.caseName(), programCtrl); timeStep < fileNameIndices.size(); timeStep++)
        {
            // We should check for field names here. Currently we are just doing the default fields
            const host::arrayCollection<scalar_t, ctorType::MUST_READ> hostMoments(
                programCtrl,
                {"rho", "u", "v", "w", "m_xx", "m_xy", "m_xz", "m_yy", "m_yz", "m_zz"},
                timeStep);

            spatialMean(hostMoments, mesh, fileNameIndices[timeStep]);
        }
    }

    if (calculationTypeString == "vorticity")
    {
        // Get the conversion type
        const name_t conversion = programCtrl.getArgument("-fileType");

        // Get the writer function
        const std::unordered_map<name_t, postProcess::writerFunction>::const_iterator it = postProcess::writers.find(conversion);

        // Get the time indices
        const std::vector<label_t> fileNameIndices = fileIO::timeIndices(programCtrl.caseName());

        if (it != postProcess::writers.end())
        {
            for (label_t timeStep = fileIO::getStartIndex(programCtrl.caseName(), programCtrl); timeStep < fileNameIndices.size(); timeStep++)
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
        const std::vector<label_t> fileNameIndices = fileIO::timeIndices(programCtrl.caseName());

        if (it != postProcess::writers.end())
        {
            for (label_t timeStep = fileIO::getStartIndex(programCtrl.caseName(), programCtrl); timeStep < fileNameIndices.size(); timeStep++)
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
            }
        }
    }

    // constexpr label_t IntegrationOrder = 2;

    // // Integrate the vorticity in all axes
    // const std::vector<scalar_t> int_omega_x = integral(integrate_x<IntegrationOrder, scalar_t>(omega[0], mesh));
    // const std::vector<scalar_t> int_omega_y = integral(integrate_y<IntegrationOrder, scalar_t>(omega[1], mesh));
    // const std::vector<scalar_t> int_omega_z = integral(integrate_z<IntegrationOrder, scalar_t>(omega[2], mesh));

    // const std::vector<std::vector<scalar_t>> integratedOmega = {int_omega_x, int_omega_y, int_omega_z};

    // // Write the files
    // // postProcess::writeVTU({magu}, "mag[u].vtu", mesh, {"mag[u]"});
    // // postProcess::writeVTU({divu}, "div[u].vtu", mesh, {"div[u]"});
    // // postProcess::writeVTU(omega, "curl[u].vtu", mesh, {"curl_x[u]", "curl_y[u]", "curl_z[u]"});
    // // postProcess::writeVTU({magomega}, "mag[curl[u]].vtu", mesh, {"mag[curl[u]]"});

    // postProcess::writeVTU(integratedOmega, "integrated_omega.vtu", mesh, {"int_omega_x", "int_omega_y", "int_omega_z"});

    return 0;
}
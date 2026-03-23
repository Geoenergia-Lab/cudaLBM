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
    Post-processing utility to convert saved moment fields to other formats
    Supported formats: VTK (.vtu) and Tecplot (.dat)

Namespace
    LBM

SourceFiles
    fieldConvert.cu

\*---------------------------------------------------------------------------*/

#include "fieldConvert.cuh"

using namespace LBM;

int main(const int argc, const char *const argv[])
{
    const programControl programCtrl(argc, argv);

    const host::latticeMesh mesh(programCtrl);

    // If we have supplied a -fieldName argument, replace programCtrl.caseName() with the fieldName
    const bool doCustomField = programCtrl.input().isArgPresent("-fieldName");
    const name_t fileNamePrefix = doCustomField ? programCtrl.getArgument("-fieldName") : programCtrl.caseName();

    // If we have supplied the -cutPlane argument, set the flag to true
    const bool doCutPlane = programCtrl.input().isArgPresent("-cutPlane");

    // Get the mesh for processing
    const host::latticeMesh newMesh = processMesh(mesh, programCtrl, doCutPlane);

    // Now get the std::vector of std::strings corresponding to the prefix
    const words_t &fieldNames = getFieldNames(fileNamePrefix, doCustomField);

    // Get the time indices
    const std::vector<host::label_t> fileNameIndices = fileIO::timeIndices(fileNamePrefix);

    // Get the conversion type
    const name_t conversion = programCtrl.getArgument("-fileType");

    // Get the writer function
    const std::unordered_map<name_t, postProcess::writerFunction>::const_iterator it = postProcess::writers.find(conversion);

    // Check if the writer is valid
    if (it != postProcess::writers.end())
    {
        const postProcess::writerFunction writer = it->second;

        for (host::label_t timeStep = fileIO::getStartIndex(fileNamePrefix, programCtrl); timeStep < fileNameIndices.size(); timeStep++)
        {
            const host::arrayCollection<scalar_t, ctorType::MUST_READ> hostMoments = initialiseArrays(
                fileNamePrefix,
                programCtrl,
                fieldNames,
                timeStep);

            const std::vector<std::vector<scalar_t>> fields = processFields(hostMoments, mesh, programCtrl, doCutPlane);

            const name_t fileName = processName(programCtrl, fileNamePrefix, fileNameIndices[timeStep], doCutPlane);

            writer(
                fields,
                fileName,
                newMesh,
                fieldNames);

            if (timeStep < fileNameIndices.size() - 1)
            {
                std::cout << std::endl;
            }
        }
    }
    else
    {
        // Throw
        throw std::runtime_error(invalidWriter(postProcess::writers, conversion));
    }

    return 0;
}
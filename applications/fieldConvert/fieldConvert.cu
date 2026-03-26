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

    // Check if is multiphase
    const bool isMultiphase = programCtrl.isMultiphase();

    // If we have supplied a -fieldName argument, replace programCtrl.caseName() with the fieldName
    const bool doCustomField = programCtrl.input().isArgPresent("-fieldName");
    const name_t fileNamePrefix = doCustomField ? programCtrl.getArgument("-fieldName") : programCtrl.caseName();

    // If we have supplied the -cutPlane argument, set the flag to true
    const bool doCutPlane = programCtrl.input().isArgPresent("-cutPlane");

    // Get the mesh for processing
    const host::latticeMesh newMesh = processMesh(mesh, programCtrl, doCutPlane);

    // Now get the std::vector of std::strings corresponding to the prefix
    const words_t &fieldNames = getFieldNames(fileNamePrefix, doCustomField, isMultiphase);

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

            // BRENO: infer correct output naming from what was actually read/produced
            const std::vector<std::string> &fullLayout =
                functionObjects::solutionVariableNames(isMultiphase);

            std::vector<std::vector<scalar_t>> fieldsOut;
            std::vector<std::string> fieldNamesOut;

            if (doCustomField)
            {
                for (const auto &requested : fieldNames)
                {
                    auto itName = std::find(fullLayout.begin(), fullLayout.end(), requested);

                    if (itName == fullLayout.end())
                    {
                        throw std::runtime_error("Requested variable not found in solution layout: " + requested);
                    }

                    const auto signedIdx = std::distance(fullLayout.begin(), itName);

                    if (signedIdx < 0)
                    {
                        throw std::runtime_error("Negative index computed while filtering fields.");
                    }

                    const std::size_t idx = static_cast<std::size_t>(signedIdx);

                    if (idx >= fields.size())
                    {
                        throw std::runtime_error("Field index out of bounds while filtering: " + requested);
                    }

                    fieldsOut.push_back(fields[idx]);
                    fieldNamesOut.push_back(requested);
                }
            }
            else
            {
                fieldsOut = fields;
                fieldNamesOut = fullLayout;
            }

            const name_t fileName = processName(programCtrl, fileNamePrefix, fileNameIndices[timeStep], doCutPlane);

            writer(
                fieldsOut,
                fileName,
                newMesh,
                fieldNamesOut);
        }
    }
    else
    {
        // Throw
        throw std::runtime_error(invalidWriter(postProcess::writers, conversion));
    }

    return 0;
}
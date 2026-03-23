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
    Aliases and kernel definitions for the isothermal D3Q19 moment
    representation lattice Boltzmann model

Namespace
    LBM

SourceFiles
    isothermalD3Q19.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_ISOTHERMALD3Q19_CUH
#define __MBLBM_ISOTHERMALD3Q19_CUH

#include "../../../src/LBMIncludes.cuh"
#include "../../../src/typedefs/typedefs.cuh"
#include "../../../src/streaming/streaming.cuh"
#include "../../../src/collision/collision.cuh"
#include "../../../src/blockHalo/blockHalo.cuh"
#include "../../../src/fileIO/fileIO.cuh"
#include "../../../src/runTimeIO/runTimeIO.cuh"
#include "../../../src/functionObjects/objectRegistry.cuh"
#include "../../../src/array/array.cuh"
#include "../../../src/boundaryConditions/boundaryConditions.cuh"

namespace LBM
{
    using BoundaryConditions = boundaryConditions::traits<boundaryConditions::caseName()>::type;
    using VelocitySet = D3Q19<Isothermal>;
    using Collision = secondOrder;
}

#include "../../../src/momentBasedLBM/kernel.cuh"

#endif
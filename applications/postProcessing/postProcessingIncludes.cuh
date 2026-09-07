/*---------------------------------------------------------------------------*\
|                                                                             |
| HermiteLBM: CUDA-based moment representation Lattice Boltzmann Method       |
| Developed at UDESC - State University of Santa Catarina                     |
| Website: https://www.udesc.br                                               |
| Github: https://github.com/Geoenergia-Lab/HermiteLBM                        |
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
    This file is part of HermiteLBM.

    HermiteLBM is free software: you can redistribute it and/or modify it
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
    Includes necessary for all post processing executables

Namespace
    LBM

SourceFiles
    postProcessingIncludes.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_POSTPROCESSINGINCLUDES_CUH
#define __MBLBM_POSTPROCESSINGINCLUDES_CUH

#include "../../src/LBMIncludes.cuh"
#include "../../src/IO/basicIO.cuh"
#include "../../src/typedefs/typedefs.cuh"
#include "../../src/array/array.cuh"
#include "../../src/fields/fields.cuh"
#include "../../src/IO/fileIO/fileIO.cuh"
#include "../../src/postProcess/postProcess.cuh"
#include "../../src/programControl/programControl.cuh"
#include "../../src/numericalSchemes/numericalSchemes.cuh"

#endif
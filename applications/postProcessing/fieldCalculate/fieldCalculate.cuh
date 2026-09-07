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
    Function definitions and includes specific to the fieldCalculate executable

Namespace
    LBM

SourceFiles
    fieldCalculate.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_FIELDCALCULATE_CUH
#define __MBLBM_FIELDCALCULATE_CUH

#include "../postProcessingIncludes.cuh"
#include "calculators.cuh"
#include "reductionCalculators.cuh"
#include "pointwiseCalculators.cuh"

namespace LBM
{
    /**
     * @brief Unordered map of the writer types to the appropriate functions
     **/
    const std::unordered_map<name_t, calculator::functionType> calculators = {
        {"containsNaN", calculator::containsNaN},
        {"mean", calculator::spatialMean},
        {"sum", calculator::spatialSum},
        {"max", calculator::fieldMax},
        {"min", calculator::fieldMin},
        {"absMax", calculator::fieldAbsMax},
        {"absMin", calculator::fieldAbsMin},
        {"magnitude", calculator::magnitude},
        {"magnitudeSquared", calculator::magnitudeSquared},
        {"dfdx", calculator::diff<axis::X>},
        {"dfdx_v2", calculator::dfdx_v2},
        {"dfdy", calculator::diff<axis::Y>},
        {"dfdz", calculator::diff<axis::Z>},
        {"div", calculator::div}};
}

#endif
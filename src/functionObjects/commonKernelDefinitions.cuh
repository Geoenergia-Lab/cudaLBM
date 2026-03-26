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
    Definitions of CUDA kernels to calculate solution quantities. Unfortunately
    we cannot template CUDA kernels and annotate with launch bounds at the same
    time due to the compiler apparently not noticing that the argument
    preceding the launch bounds is a specification of a template parameter.
    Instead, we have to do this preprocessor nonsense. We live in a cruel world.

Namespace
    LBM::functionObjects

SourceFiles
    commonKernelDefinitions.cuh

\*---------------------------------------------------------------------------*/

/**
 * @brief CUDA kernel for calculating the time averaged quantity only
 * @param[in] devPtrs Device pointer collection containing density, velocity and moment fields
 * @param[out] resultMeanPtrs Device pointer collection for the time averaged quantity
 * @param[in] invNewCount Reciprocal of (nTimeSteps + 1) for time averaging
 **/
__launch_bounds__(block::maxThreads(), This::MIN_BLOCKS_PER_MP()) __global__ void mean(
    const device::ptrCollection<10, const scalar_t> devPtrs,
    const device::ptrCollection<This::N, scalar_t> resultMeanPtrs,
    const scalar_t invNewCount)
{
    functionObjects::mean<This>(devPtrs, resultMeanPtrs, invNewCount);
}

/**
 * @brief CUDA kernel for calculating the instantaneous and time averaged quantity
 * @param[in] devPtrs Device pointer collection containing density, velocity and moment fields
 * @param[out] resulPtrs Device pointer collection for the instantaneous quantity
 * @param[out] resultMeanPtrs Device pointer collection for the time averaged quantity
 * @param[in] invNewCount Reciprocal of (nTimeSteps + 1) for time averaging
 **/
__launch_bounds__(block::maxThreads(), This::MIN_BLOCKS_PER_MP()) static __global__ void instantaneousAndMean(
    const device::ptrCollection<10, const scalar_t> devPtrs,
    const device::ptrCollection<This::N, scalar_t> resultPtrs,
    const device::ptrCollection<This::N, scalar_t> resultMeanPtrs,
    const scalar_t invNewCount)
{
    functionObjects::instantaneousAndMean<This>(devPtrs, resultPtrs, resultMeanPtrs, invNewCount);
}

/**
 * @brief CUDA kernel for calculating the instantaneous quantity only
 * @param[in] devPtrs Device pointer collection containing density, velocity and moment fields
 * @param[out] resulPtrs Device pointer collection for the instantaneous quantity
 **/
__launch_bounds__(block::maxThreads(), This::MIN_BLOCKS_PER_MP()) static __global__ void instantaneous(
    const device::ptrCollection<10, const scalar_t> devPtrs,
    const device::ptrCollection<This::N, scalar_t> resultPtrs)
{
    functionObjects::instantaneous<This>(devPtrs, resultPtrs);
}

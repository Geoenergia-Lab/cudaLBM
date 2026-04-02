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
Authors: Nathan Duggins, Breno Gemelgo (Geoenergia Lab, UDESC)

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
    Main kernels for the multiphase moment representation with the D3Q19
    velocity set for hydrodynamics and D3Q7 for phase field evolution

Namespace
    LBM

SourceFiles
    phaseFieldD3Q19.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_PHASEFIELDD3Q19_CUH
#define __MBLBM_PHASEFIELDD3Q19_CUH

#include "../../src/LBMIncludes.cuh"
#include "../../src/typedefs/typedefs.cuh"
#include "../../src/streaming/streaming.cuh"
#include "../../src/collision/collision.cuh"
#include "../../src/blockHalo/blockHalo.cuh"
#include "../../src/fileIO/fileIO.cuh"
#include "../../src/runTimeIO/runTimeIO.cuh"
#include "../../src/functionObjects/objectRegistry.cuh"
#include "../../src/array/array.cuh"
#include "../../src/boundaryConditions/boundaryConditions.cuh"
#include "../../src/momentBasedLBM/phaseKernel.cuh"

namespace LBM
{
    using BoundaryConditions = multiphase::boundaryConditions::traits<multiphase::boundaryConditions::caseName()>::type;
    using VelocitySet = D3Q19<Isothermal>;
    using PhaseVelocitySet = D3Q7;
    using Collision = secondOrder;
    using HydroHalo = device::halo<VelocitySet, BoundaryConditions::periodicX(), BoundaryConditions::periodicY(), BoundaryConditions::periodicZ()>;
    using PhaseHalo = device::halo<PhaseVelocitySet, BoundaryConditions::periodicX(), BoundaryConditions::periodicY(), BoundaryConditions::periodicZ()>;

#ifndef launchBoundsD3Q19
#define launchBoundsD3Q19 __launch_bounds__(block::maxThreads(), 2)
#endif

    /**
     * @brief Stream step wrapper for multiphase D3Q19 + D3Q7
     **/
    launchBoundsD3Q19 __global__ void phaseFieldStream(
        const device::ptrCollection<NUMBER_MOMENTS<true>(), scalar_t> devPtrs,
        const device::ptrCollection<6, const scalar_t> ghostHydro,
        const device::ptrCollection<6, const scalar_t> ghostPhase,
        const device::ptrCollection<6, const scalar_t> ghostPhi,
        const device::ptrCollection<6, scalar_t> ghostPhiWrite)
    {
        extern __shared__ scalar_t hydroShared[];
        __shared__ scalar_t phaseShared[(PhaseVelocitySet::Q() - 1) * block::stride()];

        phaseStream<BoundaryConditions, VelocitySet, PhaseVelocitySet, HydroHalo, PhaseHalo>(
            devPtrs,
            ghostHydro,
            ghostPhase,
            ghostPhi,
            ghostPhiWrite,
            hydroShared,
            phaseShared);
    }

    /**
     * @brief Collision step wrapper for multiphase D3Q19 + D3Q7
     **/
    launchBoundsD3Q19 __global__ void phaseFieldCollide(
        const device::ptrCollection<NUMBER_MOMENTS<true>(), scalar_t> devPtrs,
        const device::ptrCollection<6, scalar_t> ghostHydro,
        const device::ptrCollection<6, scalar_t> ghostPhase,
        const device::ptrCollection<6, const scalar_t> ghostPhi)
    {
        phaseCollide<BoundaryConditions, VelocitySet, PhaseVelocitySet, Collision, HydroHalo, PhaseHalo>(
            devPtrs,
            ghostHydro,
            ghostPhase,
            ghostPhi);
    }

    /**
     * @brief Prime packed scalar phi halos from the current phi field (bootstrap step)
     **/
    launchBoundsD3Q19 __global__ void phaseFieldPrimePhiHalo(
        const device::ptrCollection<NUMBER_MOMENTS<true>(), scalar_t> devPtrs,
        const device::ptrCollection<6, scalar_t> ghostPhiWrite)
    {
        const thread::coordinate Tx;
        const block::coordinate Bx;
        const device::pointCoordinate point(Tx, Bx);

        if constexpr (out_of_bounds_check())
        {
            if (device::out_of_bounds(point))
            {
                return;
            }
        }

        const device::label_t idx = device::idx(Tx, Bx);
        const scalar_t phi_ = devPtrs.ptr<10>()[idx];

        PhaseHalo::save_scalar(phi_, ghostPhiWrite, Tx, Bx, point);
    }
}

#endif

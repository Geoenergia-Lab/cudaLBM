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
    A class holding information about the solution grid

Namespace
    LBM::host

SourceFiles
    latticeMesh.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_LATTICEMESH_CUH
#define __MBLBM_LATTICEMESH_CUH

#include "../LBMIncludes.cuh"
#include "../typedefs/typedefs.cuh"
#include "../globalFunctions.cuh"
#include "../programControl/programControl.cuh"

#include "hostLatticeMesh.cuh"
#include "deviceLatticeMesh.cuh"

namespace LBM
{
    namespace host
    {
        /**
         * @brief Memory index (host version)
         * @param[in] tx,ty,tz Thread-local coordinates
         * @param[in] bx,by,bz Block indices
         * @param[in] mesh The lattice mesh
         * @return Linearized index using mesh constants
         *
         * Layout: [bx][by][bz][tz][ty][tx] (tx fastest varying)
         **/
        __host__ [[nodiscard]] inline label_t idx(
            const label_t tx, const label_t ty, const label_t tz,
            const label_t bx, const label_t by, const label_t bz,
            const latticeMesh &mesh) noexcept
        {
            return idx(tx, ty, tz, bx, by, bz, mesh.nBlocks<axis::X>(), mesh.nBlocks<axis::Y>());
        }

        __host__ [[nodiscard]] inline label_t idx(
            const threadLabel &Tx,
            const blockLabel &Bx,
            const latticeMesh &mesh) noexcept
        {
            return idx(Tx.x, Tx.y, Tx.z, Bx.x, Bx.y, Bx.z, mesh.nBlocks<axis::X>(), mesh.nBlocks<axis::Y>());
        }
    }
}

#endif
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
Authors: Nathan Duggins, Vinicius Czarnobay, Breno Gemelgo (Geoenergia Lab, UDESC)

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
    Header file to avoid repeated definition of the boundary condition function

SourceFiles
    jetBoundaryCondition.cuh

Notes
    This file is intended to be included directly inside a switch-case block.
    Do NOT use include guards (#ifndef/#define/#endif).

\*---------------------------------------------------------------------------*/

assertions::velocitySet::validate<VelocitySet>();

const scalar_t rho_I = velocitySet::calculate_moment<VelocitySet, NO_DIRECTION, NO_DIRECTION>(pop, boundaryNormal);
const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;

switch (boundaryNormal.nodeType())
{
// Round inflow + no-slip
case normalVector::BACK():
{
    // MODIFY THIS FOR MULTI GPU
    const label_t x = threadIdx.x + block::nx() * blockIdx.x;
    const label_t y = threadIdx.y + block::ny() * blockIdx.y;

    const scalar_t dx = static_cast<scalar_t>(x) - center_x();
    const scalar_t dy = static_cast<scalar_t>(y) - center_y();

    if constexpr (new_inlet())
    {
        const scalar_t is_jet = static_cast<scalar_t>(((dx * dx) + (dy * dy)) < r2());
        const scalar_t rho = rho0<scalar_t>();
        moments[m_i<0>()] = rho;                                                    // rho
        moments[m_i<1>()] = is_jet * device::U_Back[0];                             // ux
        moments[m_i<2>()] = is_jet * device::U_Back[1];                             // uy
        moments[m_i<3>()] = is_jet * device::U_Back[2];                             // uz
        moments[m_i<4>()] = is_jet * (rho * device::U_Back[0] * device::U_Back[0]); // mxx
        moments[m_i<5>()] = is_jet * (rho * device::U_Back[0] * device::U_Back[1]); // mxy
        moments[m_i<6>()] = is_jet * (rho * device::U_Back[0] * device::U_Back[2]); // mxz
        moments[m_i<7>()] = is_jet * (rho * device::U_Back[1] * device::U_Back[1]); // myy
        moments[m_i<8>()] = is_jet * (rho * device::U_Back[1] * device::U_Back[2]); // myz
        moments[m_i<9>()] = is_jet * (rho * device::U_Back[2] * device::U_Back[2]); // mzz
    }
    else
    {
        const scalar_t is_jet = static_cast<scalar_t>((static_cast<scalar_t>(x) - center_x()) * (static_cast<scalar_t>(x) - center_x()) + (static_cast<scalar_t>(y) - center_y()) * (static_cast<scalar_t>(y) - center_y()) < r2());
        const scalar_t mxz_I = velocitySet::calculate_moment<VelocitySet, X, Z>(pop, boundaryNormal) * inv_rho_I;
        const scalar_t myz_I = velocitySet::calculate_moment<VelocitySet, Y, Z>(pop, boundaryNormal) * inv_rho_I;

        const scalar_t rho = rho0<scalar_t>();
        const scalar_t mxz = static_cast<scalar_t>(2) * mxz_I * rho_I / rho;
        const scalar_t myz = static_cast<scalar_t>(2) * myz_I * rho_I / rho;

        moments[m_i<0>()] = rho;                                                    // rho
        moments[m_i<1>()] = is_jet * device::U_Back[0];                             // ux
        moments[m_i<2>()] = is_jet * device::U_Back[1];                             // uy
        moments[m_i<3>()] = is_jet * device::U_Back[2];                             // uz
        moments[m_i<4>()] = static_cast<scalar_t>(0);                               // mxx
        moments[m_i<5>()] = static_cast<scalar_t>(0);                               // mxy
        moments[m_i<6>()] = mxz;                                                    // mxz
        moments[m_i<7>()] = static_cast<scalar_t>(0);                               // myy
        moments[m_i<8>()] = myz;                                                    // myz
        moments[m_i<9>()] = is_jet * (rho * device::U_Back[2] * device::U_Back[2]); // mzz
    }

    return;
}

// Lateral boundary faces and edges
#include "include/lateralFacesAndEdges.cuh"

// Outlet face, edges and corners
#include "include/outlet.cuh"

// Edges and corners on the inlet plane
#include "include/inletEdgesAndCorners.cuh"
}
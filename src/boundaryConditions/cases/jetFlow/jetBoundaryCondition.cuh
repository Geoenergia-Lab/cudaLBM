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

if (!(boundaryNormal.isBack() || boundaryNormal.isFront()))
{
    // moments[m_i<0>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + m_i<0>()];
    moments[m_i<0>()] = rho0();
    return;
}

const scalar_t rho_I = velocitySet::calculate_moment<VelocitySet, axis::NO_DIRECTION, axis::NO_DIRECTION>(pop, boundaryNormal);
const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;
const device::label_t tid = block::idx(Tx.value<axis::X>(), Tx.value<axis::Y>(), block::nz() - 2);

const scalar_t is_jet = static_cast<scalar_t>(boundaryNormal.isBack() && rms_sq(point.value<axis::X, scalar_t>() - center_x(), point.value<axis::Y, scalar_t>() - center_y()) <= r2());

const scalar_t is_outlet = static_cast<scalar_t>(boundaryNormal.isFront());

moments[m_i<0>()] = (is_outlet * shared_buffer[tid * (NUMBER_MOMENTS() + 1) + m_i<0>()]);
moments[m_i<1>()] = (is_outlet * shared_buffer[tid * (NUMBER_MOMENTS() + 1) + m_i<1>()]) + (is_jet * device::U_Back[0]);
moments[m_i<2>()] = (is_outlet * shared_buffer[tid * (NUMBER_MOMENTS() + 1) + m_i<2>()]) + (is_jet * device::U_Back[1]);
moments[m_i<3>()] = (is_outlet * shared_buffer[tid * (NUMBER_MOMENTS() + 1) + m_i<3>()]) + (is_jet * device::U_Back[2]);

// Set equilibrium velocities
moments[m_i<4>()] = moments[m_i<1>()] * moments[m_i<1>()];
moments[m_i<5>()] = moments[m_i<1>()] * moments[m_i<2>()];
moments[m_i<6>()] = moments[m_i<1>()] * moments[m_i<3>()];
moments[m_i<7>()] = moments[m_i<2>()] * moments[m_i<2>()];
moments[m_i<8>()] = moments[m_i<2>()] * moments[m_i<3>()];
moments[m_i<9>()] = moments[m_i<3>()] * moments[m_i<3>()];

switch (boundaryNormal.nodeType())
{
// Round inflow + no-slip
case normalVector::BACK():
{
    if constexpr (new_inlet())
    {
        // const scalar_t is_jet = static_cast<scalar_t>(rms_sq(point.value<axis::X, scalar_t>() - center_x(), point.value<axis::Y, scalar_t>() - center_y()) <= r2());

        const scalar_t mxz_I = velocitySet::calculate_moment<VelocitySet, axis::X, axis::Z>(pop, boundaryNormal) * inv_rho_I;
        const scalar_t myz_I = velocitySet::calculate_moment<VelocitySet, axis::Y, axis::Z>(pop, boundaryNormal) * inv_rho_I;

        // const scalar_t A = static_cast<scalar_t>(3) * (device::U_Back[2] * (device::U_Back[2] * device::U_Back[2]));

        // Density
        // moments[m_i<0>()] = (static_cast<scalar_t>(6) * rho_I) / (static_cast<scalar_t>(-5) + (A * is_jet));
        // moments[m_i<0>()] = rho0();
        // moments[m_i<0>()] = rho0() + ((static_cast<scalar_t>(6) * rho_I) / (static_cast<scalar_t>(-5) + (A * is_jet))); // rho

        // Now try this if stable
        moments[m_i<0>()] = shared_buffer[block::idx(Tx.value<axis::X>(), Tx.value<axis::Y>(), 1) * (NUMBER_MOMENTS() + 1) + m_i<0>()];

        // Velocity
        // moments[m_i<1>()] = is_jet * device::U_Back[0]; // ux
        // moments[m_i<2>()] = is_jet * device::U_Back[1]; // uy
        // moments[m_i<3>()] = is_jet * device::U_Back[2]; // uz

        // Moments
        // moments[m_i<4>()] = static_cast<scalar_t>(0);
        // moments[m_i<5>()] = static_cast<scalar_t>(0);
        // moments[m_i<6>()] = ((static_cast<scalar_t>(5) * mxz_I) - (A * mxz_I)) / static_cast<scalar_t>(3);
        moments[m_i<6>()] = static_cast<scalar_t>(2) * mxz_I * rho_I / moments[m_i<0>()];
        // moments[m_i<7>()] = static_cast<scalar_t>(0);
        // moments[m_i<8>()] = ((static_cast<scalar_t>(5) * myz_I) - (A * myz_I)) / static_cast<scalar_t>(3);
        moments[m_i<8>()] = static_cast<scalar_t>(2) * myz_I * rho_I / moments[m_i<0>()];
        // moments[m_i<9>()] = is_jet * (moments[m_i<0>()] * device::U_Back[2] * device::U_Back[2]);
    }
    else
    {
        // const scalar_t is_jet = static_cast<scalar_t>((static_cast<scalar_t>(point.value<axis::X>()) - center_x()) * (static_cast<scalar_t>(point.value<axis::X>()) - center_x()) + (static_cast<scalar_t>(point.value<axis::Y>()) - center_y()) * (static_cast<scalar_t>(point.value<axis::Y>()) - center_y()) < r2());
        const scalar_t mxz_I = velocitySet::calculate_moment<VelocitySet, axis::X, axis::Z>(pop, boundaryNormal) * inv_rho_I;
        const scalar_t myz_I = velocitySet::calculate_moment<VelocitySet, axis::Y, axis::Z>(pop, boundaryNormal) * inv_rho_I;

        const scalar_t rho = rho0();
        const scalar_t mxz = static_cast<scalar_t>(2) * mxz_I * rho_I / rho;
        const scalar_t myz = static_cast<scalar_t>(2) * myz_I * rho_I / rho;

        moments[m_i<0>()] = rho; // rho
        // moments[m_i<1>()] = is_jet * device::U_Back[0];                             // ux
        // moments[m_i<2>()] = is_jet * device::U_Back[1];                             // uy
        // moments[m_i<3>()] = is_jet * device::U_Back[2];                             // uz
        // moments[m_i<4>()] = static_cast<scalar_t>(0);                               // mxx
        // moments[m_i<5>()] = static_cast<scalar_t>(0);                               // mxy
        // moments[m_i<6>()] = mxz;                                                    // mxz
        // moments[m_i<7>()] = static_cast<scalar_t>(0);                               // myy
        // moments[m_i<8>()] = myz;                                                    // myz
        // moments[m_i<9>()] = is_jet * (rho * device::U_Back[2] * device::U_Back[2]); // mzz
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
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
    Applies IRBC Neumann-type boundary conditions

SourceFiles
    IRBCNeumann.cuh

Notes
    This file is intended to be included directly inside a switch-case block.
    Do NOT use include guards (#ifndef/#define/#endif).

\*---------------------------------------------------------------------------*/

// Corners
case normalVector::WEST_SOUTH_FRONT():
{
    const label_t tid = device::idxBlock(threadIdx.x, threadIdx.y, threadIdx.z - 1);

    // Classic Neumann
    moments[m_i<0>()] = shared_buffer[tid * (NUMBER_MOMENTS<false>() + 1) + m_i<0>()]; // p
    moments[m_i<1>()] = shared_buffer[tid * (NUMBER_MOMENTS<false>() + 1) + m_i<1>()]; // ux
    moments[m_i<2>()] = shared_buffer[tid * (NUMBER_MOMENTS<false>() + 1) + m_i<2>()]; // uy
    moments[m_i<3>()] = shared_buffer[tid * (NUMBER_MOMENTS<false>() + 1) + m_i<3>()]; // uz

    // IRBC-Neumann
    moments[m_i<4>()] = moments[m_i<1>()] * moments[m_i<1>()]; // mxx
    moments[m_i<5>()] = moments[m_i<1>()] * moments[m_i<2>()]; // mxy
    moments[m_i<6>()] = moments[m_i<1>()] * moments[m_i<3>()]; // mxz
    moments[m_i<7>()] = moments[m_i<2>()] * moments[m_i<2>()]; // myy
    moments[m_i<8>()] = moments[m_i<2>()] * moments[m_i<3>()]; // myz
    moments[m_i<9>()] = moments[m_i<3>()] * moments[m_i<3>()]; // mzz

    return;
}
case normalVector::WEST_NORTH_FRONT():
{
    const label_t tid = device::idxBlock(threadIdx.x, threadIdx.y, threadIdx.z - 1);

    // Classic Neumann
    moments[m_i<0>()] = shared_buffer[tid * (NUMBER_MOMENTS<false>() + 1) + m_i<0>()]; // p
    moments[m_i<1>()] = shared_buffer[tid * (NUMBER_MOMENTS<false>() + 1) + m_i<1>()]; // ux
    moments[m_i<2>()] = shared_buffer[tid * (NUMBER_MOMENTS<false>() + 1) + m_i<2>()]; // uy
    moments[m_i<3>()] = shared_buffer[tid * (NUMBER_MOMENTS<false>() + 1) + m_i<3>()]; // uz

    // IRBC-Neumann
    moments[m_i<4>()] = moments[m_i<1>()] * moments[m_i<1>()]; // mxx
    moments[m_i<5>()] = moments[m_i<1>()] * moments[m_i<2>()]; // mxy
    moments[m_i<6>()] = moments[m_i<1>()] * moments[m_i<3>()]; // mxz
    moments[m_i<7>()] = moments[m_i<2>()] * moments[m_i<2>()]; // myy
    moments[m_i<8>()] = moments[m_i<2>()] * moments[m_i<3>()]; // myz
    moments[m_i<9>()] = moments[m_i<3>()] * moments[m_i<3>()]; // mzz

    return;
}
case normalVector::EAST_SOUTH_FRONT():
{
    const label_t tid = device::idxBlock(threadIdx.x, threadIdx.y, threadIdx.z - 1);

    // Classic Neumann
    moments[m_i<0>()] = shared_buffer[tid * (NUMBER_MOMENTS<false>() + 1) + m_i<0>()]; // p
    moments[m_i<1>()] = shared_buffer[tid * (NUMBER_MOMENTS<false>() + 1) + m_i<1>()]; // ux
    moments[m_i<2>()] = shared_buffer[tid * (NUMBER_MOMENTS<false>() + 1) + m_i<2>()]; // uy
    moments[m_i<3>()] = shared_buffer[tid * (NUMBER_MOMENTS<false>() + 1) + m_i<3>()]; // uz

    // IRBC-Neumann
    moments[m_i<4>()] = moments[m_i<1>()] * moments[m_i<1>()]; // mxx
    moments[m_i<5>()] = moments[m_i<1>()] * moments[m_i<2>()]; // mxy
    moments[m_i<6>()] = moments[m_i<1>()] * moments[m_i<3>()]; // mxz
    moments[m_i<7>()] = moments[m_i<2>()] * moments[m_i<2>()]; // myy
    moments[m_i<8>()] = moments[m_i<2>()] * moments[m_i<3>()]; // myz
    moments[m_i<9>()] = moments[m_i<3>()] * moments[m_i<3>()]; // mzz

    return;
}
case normalVector::EAST_NORTH_FRONT():
{
    const label_t tid = device::idxBlock(threadIdx.x, threadIdx.y, threadIdx.z - 1);

    // Classic Neumann
    moments[m_i<0>()] = shared_buffer[tid * (NUMBER_MOMENTS<false>() + 1) + m_i<0>()]; // p
    moments[m_i<1>()] = shared_buffer[tid * (NUMBER_MOMENTS<false>() + 1) + m_i<1>()]; // ux
    moments[m_i<2>()] = shared_buffer[tid * (NUMBER_MOMENTS<false>() + 1) + m_i<2>()]; // uy
    moments[m_i<3>()] = shared_buffer[tid * (NUMBER_MOMENTS<false>() + 1) + m_i<3>()]; // uz

    // IRBC-Neumann
    moments[m_i<4>()] = moments[m_i<1>()] * moments[m_i<1>()]; // mxx
    moments[m_i<5>()] = moments[m_i<1>()] * moments[m_i<2>()]; // mxy
    moments[m_i<6>()] = moments[m_i<1>()] * moments[m_i<3>()]; // mxz
    moments[m_i<7>()] = moments[m_i<2>()] * moments[m_i<2>()]; // myy
    moments[m_i<8>()] = moments[m_i<2>()] * moments[m_i<3>()]; // myz
    moments[m_i<9>()] = moments[m_i<3>()] * moments[m_i<3>()]; // mzz

    return;
}

// Edges
case normalVector::WEST_FRONT():
{
    const label_t tid = device::idxBlock(threadIdx.x, threadIdx.y, threadIdx.z - 1);

    // Classic Neumann
    moments[m_i<0>()] = shared_buffer[tid * (NUMBER_MOMENTS<false>() + 1) + m_i<0>()]; // p
    moments[m_i<1>()] = shared_buffer[tid * (NUMBER_MOMENTS<false>() + 1) + m_i<1>()]; // ux
    moments[m_i<2>()] = shared_buffer[tid * (NUMBER_MOMENTS<false>() + 1) + m_i<2>()]; // uy
    moments[m_i<3>()] = shared_buffer[tid * (NUMBER_MOMENTS<false>() + 1) + m_i<3>()]; // uz

    // Incoming moments
    const scalar_t mxy_I = velocitySet::calculate_moment<VelocitySet, X, Y>(pop, boundaryNormal);
    const scalar_t myz_I = velocitySet::calculate_moment<VelocitySet, Y, Z>(pop, boundaryNormal);

    if constexpr (VelocitySet::Q() == 19)
    {
        // IRBC-Neumann
        moments[m_i<4>()] = moments[m_i<1>()] * moments[m_i<1>()];                                               // mxx
        moments[m_i<5>()] = (static_cast<scalar_t>(6) * mxy_I + moments[m_i<2>()]) / (static_cast<scalar_t>(3)); // mxy
        moments[m_i<6>()] = moments[m_i<1>()] * moments[m_i<3>()];                                               // mxz
        moments[m_i<7>()] = moments[m_i<2>()] * moments[m_i<2>()];                                               // myy
        moments[m_i<8>()] = (static_cast<scalar_t>(6) * myz_I - moments[m_i<2>()]) / (static_cast<scalar_t>(3)); // myz
        moments[m_i<9>()] = moments[m_i<3>()] * moments[m_i<3>()];                                               // mzz
    }
    else
    {
        // IRBC-Neumann
        moments[m_i<4>()] = moments[m_i<1>()] * moments[m_i<1>()];                                                                                                                 // mxx
        moments[m_i<5>()] = -(-static_cast<scalar_t>(45) * mxy_I - static_cast<scalar_t>(9) * myz_I - static_cast<scalar_t>(5) * moments[m_i<2>()]) / (static_cast<scalar_t>(18)); // mxy
        moments[m_i<6>()] = moments[m_i<1>()] * moments[m_i<3>()];                                                                                                                 // mxz
        moments[m_i<7>()] = moments[m_i<2>()] * moments[m_i<2>()];                                                                                                                 // myy
        moments[m_i<8>()] = -(-static_cast<scalar_t>(9) * mxy_I - static_cast<scalar_t>(45) * myz_I + static_cast<scalar_t>(5) * moments[m_i<2>()]) / (static_cast<scalar_t>(18)); // myz
        moments[m_i<9>()] = moments[m_i<3>()] * moments[m_i<3>()];                                                                                                                 // mzz
    }

    return;
}
case normalVector::EAST_FRONT():
{
    const label_t tid = device::idxBlock(threadIdx.x, threadIdx.y, threadIdx.z - 1);

    // Classic Neumann
    moments[m_i<0>()] = shared_buffer[tid * (NUMBER_MOMENTS<false>() + 1) + m_i<0>()]; // p
    moments[m_i<1>()] = shared_buffer[tid * (NUMBER_MOMENTS<false>() + 1) + m_i<1>()]; // ux
    moments[m_i<2>()] = shared_buffer[tid * (NUMBER_MOMENTS<false>() + 1) + m_i<2>()]; // uy
    moments[m_i<3>()] = shared_buffer[tid * (NUMBER_MOMENTS<false>() + 1) + m_i<3>()]; // uz

    // Incoming moments
    const scalar_t mxy_I = velocitySet::calculate_moment<VelocitySet, X, Y>(pop, boundaryNormal);
    const scalar_t myz_I = velocitySet::calculate_moment<VelocitySet, Y, Z>(pop, boundaryNormal);

    if constexpr (VelocitySet::Q() == 19)
    {
        // IRBC-Neumann
        moments[m_i<4>()] = moments[m_i<1>()] * moments[m_i<1>()];                                               // mxx
        moments[m_i<5>()] = (static_cast<scalar_t>(6) * mxy_I - moments[m_i<2>()]) / (static_cast<scalar_t>(3)); // mxy
        moments[m_i<6>()] = moments[m_i<1>()] * moments[m_i<3>()];                                               // mxz
        moments[m_i<7>()] = moments[m_i<2>()] * moments[m_i<2>()];                                               // myy
        moments[m_i<8>()] = (static_cast<scalar_t>(6) * myz_I - moments[m_i<2>()]) / (static_cast<scalar_t>(3)); // myz
        moments[m_i<9>()] = moments[m_i<3>()] * moments[m_i<3>()];                                               // mzz
    }
    else
    {
        // IRBC-Neumann
        moments[m_i<4>()] = moments[m_i<1>()] * moments[m_i<1>()];                                                                                                                 // mxx
        moments[m_i<5>()] = -(-static_cast<scalar_t>(45) * mxy_I + static_cast<scalar_t>(9) * myz_I + static_cast<scalar_t>(5) * moments[m_i<2>()]) / (static_cast<scalar_t>(18)); // mxy
        moments[m_i<6>()] = moments[m_i<1>()] * moments[m_i<3>()];                                                                                                                 // mxz
        moments[m_i<7>()] = moments[m_i<2>()] * moments[m_i<2>()];                                                                                                                 // myy
        moments[m_i<8>()] = -(static_cast<scalar_t>(9) * mxy_I - static_cast<scalar_t>(45) * myz_I + static_cast<scalar_t>(5) * moments[m_i<2>()]) / (static_cast<scalar_t>(18));  // myz
        moments[m_i<9>()] = moments[m_i<3>()] * moments[m_i<3>()];                                                                                                                 // mzz
    }

    return;
}
case normalVector::SOUTH_FRONT():
{
    const label_t tid = device::idxBlock(threadIdx.x, threadIdx.y, threadIdx.z - 1);

    // Classic Neumann
    moments[m_i<0>()] = shared_buffer[tid * (NUMBER_MOMENTS<false>() + 1) + m_i<0>()]; // p
    moments[m_i<1>()] = shared_buffer[tid * (NUMBER_MOMENTS<false>() + 1) + m_i<1>()]; // ux
    moments[m_i<2>()] = shared_buffer[tid * (NUMBER_MOMENTS<false>() + 1) + m_i<2>()]; // uy
    moments[m_i<3>()] = shared_buffer[tid * (NUMBER_MOMENTS<false>() + 1) + m_i<3>()]; // uz

    // Incoming moments
    const scalar_t mxy_I = velocitySet::calculate_moment<VelocitySet, X, Y>(pop, boundaryNormal);
    const scalar_t mxz_I = velocitySet::calculate_moment<VelocitySet, X, Z>(pop, boundaryNormal);

    if constexpr (VelocitySet::Q() == 19)
    {
        // IRBC-Neumann
        moments[m_i<4>()] = moments[m_i<1>()] * moments[m_i<1>()];                                               // mxx
        moments[m_i<5>()] = (static_cast<scalar_t>(6) * mxy_I + moments[m_i<1>()]) / (static_cast<scalar_t>(3)); // mxy
        moments[m_i<6>()] = (static_cast<scalar_t>(6) * mxz_I - moments[m_i<1>()]) / (static_cast<scalar_t>(3)); // mxz
        moments[m_i<7>()] = moments[m_i<2>()] * moments[m_i<2>()];                                               // myy
        moments[m_i<8>()] = moments[m_i<2>()] * moments[m_i<3>()];                                               // myz
        moments[m_i<9>()] = moments[m_i<3>()] * moments[m_i<3>()];                                               // mzz
    }
    else
    {
        // IRBC-Neumann
        moments[m_i<4>()] = moments[m_i<1>()] * moments[m_i<1>()];                                                                                                                 // mxx
        moments[m_i<5>()] = -(-static_cast<scalar_t>(45) * mxy_I - static_cast<scalar_t>(9) * mxz_I - static_cast<scalar_t>(5) * moments[m_i<1>()]) / (static_cast<scalar_t>(18)); // mxy
        moments[m_i<6>()] = -(-static_cast<scalar_t>(9) * mxy_I - static_cast<scalar_t>(45) * mxz_I + static_cast<scalar_t>(5) * moments[m_i<1>()]) / (static_cast<scalar_t>(18)); // mxz
        moments[m_i<7>()] = moments[m_i<2>()] * moments[m_i<2>()];                                                                                                                 // myy
        moments[m_i<8>()] = moments[m_i<2>()] * moments[m_i<3>()];                                                                                                                 // myz
        moments[m_i<9>()] = moments[m_i<3>()] * moments[m_i<3>()];                                                                                                                 // mzz
    }

    return;
}
case normalVector::NORTH_FRONT():
{
    const label_t tid = device::idxBlock(threadIdx.x, threadIdx.y, threadIdx.z - 1);

    // Classic Neumann
    moments[m_i<0>()] = shared_buffer[tid * (NUMBER_MOMENTS<false>() + 1) + m_i<0>()]; // p
    moments[m_i<1>()] = shared_buffer[tid * (NUMBER_MOMENTS<false>() + 1) + m_i<1>()]; // ux
    moments[m_i<2>()] = shared_buffer[tid * (NUMBER_MOMENTS<false>() + 1) + m_i<2>()]; // uy
    moments[m_i<3>()] = shared_buffer[tid * (NUMBER_MOMENTS<false>() + 1) + m_i<3>()]; // uz

    // Incoming moments
    const scalar_t mxy_I = velocitySet::calculate_moment<VelocitySet, X, Y>(pop, boundaryNormal);
    const scalar_t mxz_I = velocitySet::calculate_moment<VelocitySet, X, Z>(pop, boundaryNormal);

    if constexpr (VelocitySet::Q() == 19)
    {
        // IRBC-Neumann
        moments[m_i<4>()] = moments[m_i<1>()] * moments[m_i<1>()];                                               // mxx
        moments[m_i<5>()] = (static_cast<scalar_t>(6) * mxy_I - moments[m_i<1>()]) / (static_cast<scalar_t>(3)); // mxy
        moments[m_i<6>()] = (static_cast<scalar_t>(6) * mxz_I - moments[m_i<1>()]) / (static_cast<scalar_t>(3)); // mxz
        moments[m_i<7>()] = moments[m_i<2>()] * moments[m_i<2>()];                                               // myy
        moments[m_i<8>()] = moments[m_i<2>()] * moments[m_i<3>()];                                               // myz
        moments[m_i<9>()] = moments[m_i<3>()] * moments[m_i<3>()];                                               // mzz
    }
    else
    {
        // IRBC-Neumann
        moments[m_i<4>()] = moments[m_i<1>()] * moments[m_i<1>()];                                                                                                                 // mxx
        moments[m_i<5>()] = -(-static_cast<scalar_t>(45) * mxy_I + static_cast<scalar_t>(9) * mxz_I + static_cast<scalar_t>(5) * moments[m_i<1>()]) / (static_cast<scalar_t>(18)); // mxy
        moments[m_i<6>()] = -(static_cast<scalar_t>(9) * mxy_I - static_cast<scalar_t>(45) * mxz_I + static_cast<scalar_t>(5) * moments[m_i<1>()]) / (static_cast<scalar_t>(18));  // mxz
        moments[m_i<7>()] = moments[m_i<2>()] * moments[m_i<2>()];                                                                                                                 // myy
        moments[m_i<8>()] = moments[m_i<2>()] * moments[m_i<3>()];                                                                                                                 // myz
        moments[m_i<9>()] = moments[m_i<3>()] * moments[m_i<3>()];                                                                                                                 // mzz
    }

    return;
}

// Faces
case normalVector::FRONT():
{
    const label_t tid = device::idxBlock(threadIdx.x, threadIdx.y, threadIdx.z - 1);

    // Classic Neumann
    moments[m_i<0>()] = shared_buffer[tid * (NUMBER_MOMENTS<false>() + 1) + m_i<0>()]; // p
    moments[m_i<1>()] = shared_buffer[tid * (NUMBER_MOMENTS<false>() + 1) + m_i<1>()]; // ux
    moments[m_i<2>()] = shared_buffer[tid * (NUMBER_MOMENTS<false>() + 1) + m_i<2>()]; // uy
    moments[m_i<3>()] = shared_buffer[tid * (NUMBER_MOMENTS<false>() + 1) + m_i<3>()]; // uz

    // Incoming moments
    const scalar_t mxx_I = velocitySet::calculate_moment<VelocitySet, X, X>(pop, boundaryNormal);
    const scalar_t mxy_I = velocitySet::calculate_moment<VelocitySet, X, Y>(pop, boundaryNormal);
    const scalar_t mxz_I = velocitySet::calculate_moment<VelocitySet, X, Z>(pop, boundaryNormal);
    const scalar_t myy_I = velocitySet::calculate_moment<VelocitySet, Y, Y>(pop, boundaryNormal);
    const scalar_t myz_I = velocitySet::calculate_moment<VelocitySet, Y, Z>(pop, boundaryNormal);

    if constexpr (VelocitySet::Q() == 19)
    {
        // IRBC-Neumann
        moments[m_i<4>()] = -(-static_cast<scalar_t>(4) * mxx_I + static_cast<scalar_t>(4) * myy_I - static_cast<scalar_t>(3) * moments[m_i<1>()] * moments[m_i<1>()] - static_cast<scalar_t>(3) * moments[m_i<2>()] * moments[m_i<2>()]) / (static_cast<scalar_t>(6)); // mxx
        moments[m_i<5>()] = mxy_I;                                                                                                                                                                                                                                      // mxy
        moments[m_i<6>()] = -(-static_cast<scalar_t>(6) * mxz_I + moments[m_i<1>()]) / (static_cast<scalar_t>(3));                                                                                                                                                      // mxz
        moments[m_i<7>()] = -(static_cast<scalar_t>(4) * mxx_I - static_cast<scalar_t>(4) * myy_I - static_cast<scalar_t>(3) * moments[m_i<1>()] * moments[m_i<1>()] - static_cast<scalar_t>(3) * moments[m_i<2>()] * moments[m_i<2>()]) / (static_cast<scalar_t>(6));  // myy
        moments[m_i<8>()] = -(-static_cast<scalar_t>(6) * myz_I + moments[m_i<2>()]) / (static_cast<scalar_t>(3));                                                                                                                                                      // myz
        moments[m_i<9>()] = moments[m_i<3>()] * moments[m_i<3>()];                                                                                                                                                                                                      // mzz
    }
    else
    {
        // IRBC-Neumann
        moments[m_i<4>()] = -(-static_cast<scalar_t>(6) * mxx_I + static_cast<scalar_t>(6) * myy_I - static_cast<scalar_t>(5) * moments[m_i<1>()] * moments[m_i<1>()] - static_cast<scalar_t>(5) * moments[m_i<2>()] * moments[m_i<2>()]) / (static_cast<scalar_t>(10)); // mxx
        moments[m_i<5>()] = (static_cast<scalar_t>(6) * mxy_I) / (static_cast<scalar_t>(5));                                                                                                                                                                             // mxy
        moments[m_i<6>()] = -(-static_cast<scalar_t>(6) * mxz_I + moments[m_i<1>()]) / (static_cast<scalar_t>(3));                                                                                                                                                       // mxz
        moments[m_i<7>()] = -(static_cast<scalar_t>(6) * mxx_I - static_cast<scalar_t>(6) * myy_I - static_cast<scalar_t>(5) * moments[m_i<1>()] * moments[m_i<1>()] - static_cast<scalar_t>(5) * moments[m_i<2>()] * moments[m_i<2>()]) / (static_cast<scalar_t>(10));  // myy
        moments[m_i<8>()] = -(-static_cast<scalar_t>(6) * myz_I + moments[m_i<2>()]) / (static_cast<scalar_t>(3));                                                                                                                                                       // myz
        moments[m_i<9>()] = moments[m_i<3>()] * moments[m_i<3>()];                                                                                                                                                                                                       // mzz
    }

    return;
}
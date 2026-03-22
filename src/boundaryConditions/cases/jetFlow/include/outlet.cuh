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
    Edge and corner definitions along the outlet plane of the jet.

SourceFiles
    outlet.cuh

Notes
    This file is intended to be included directly inside a switch-case block.
    Do NOT use include guards (#ifndef/#define/#endif).

\*---------------------------------------------------------------------------*/

// Faces
case normalVector::FRONT():
{
    // Classic Neumann
    // const scalar_t moments[m_i<0>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + m_i<0>()];
    // moments[m_i<0>()] = moments[m_i<0>()];
    // moments[m_i<1>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + m_i<1>()];
    // moments[m_i<2>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + m_i<2>()];
    // moments[m_i<3>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + m_i<3>()];

    if constexpr (VelocitySet::Q() == 19)
    {
        const scalar_t mxx_I = (pop[q_i<1>()] + pop[q_i<2>()] + pop[q_i<7>()] + pop[q_i<8>()] + pop[q_i<9>()] + pop[q_i<13>()] + pop[q_i<14>()] + pop[q_i<16>()]) * inv_rho_I - velocitySet::cs2<scalar_t>();
        const scalar_t mxy_I = (pop[q_i<7>()] + pop[q_i<8>()] - pop[q_i<13>()] - pop[q_i<14>()]) * inv_rho_I;
        const scalar_t mxz_I = (pop[q_i<9>()] - pop[q_i<16>()]) * inv_rho_I;
        const scalar_t myy_I = (pop[q_i<3>()] + pop[q_i<4>()] + pop[q_i<7>()] + pop[q_i<8>()] + pop[q_i<11>()] + pop[q_i<13>()] + pop[q_i<14>()] + pop[q_i<18>()]) * inv_rho_I - velocitySet::cs2<scalar_t>();
        const scalar_t myz_I = (pop[q_i<11>()] - pop[q_i<18>()]) * inv_rho_I;

        // IRBC-Neumann
        moments[m_i<4>()] = -(-static_cast<scalar_t>(4) * mxx_I * rho_I + static_cast<scalar_t>(4) * myy_I * rho_I - static_cast<scalar_t>(3) * moments[m_i<1>()] * moments[m_i<1>()] * moments[m_i<0>()] - static_cast<scalar_t>(3) * moments[m_i<2>()] * moments[m_i<2>()] * moments[m_i<0>()]) / (static_cast<scalar_t>(6) * moments[m_i<0>()]); // mxx
        moments[m_i<5>()] = (mxy_I * rho_I) / moments[m_i<0>()];                                                                                                                                                                                                                                                                                    // mxy
        moments[m_i<6>()] = -(-static_cast<scalar_t>(6) * mxz_I * rho_I + moments[m_i<1>()] * moments[m_i<0>()]) / (static_cast<scalar_t>(3) * moments[m_i<0>()]);                                                                                                                                                                                  // mxz
        moments[m_i<7>()] = -(static_cast<scalar_t>(4) * mxx_I * rho_I - static_cast<scalar_t>(4) * myy_I * rho_I - static_cast<scalar_t>(3) * moments[m_i<1>()] * moments[m_i<1>()] * moments[m_i<0>()] - static_cast<scalar_t>(3) * moments[m_i<2>()] * moments[m_i<2>()] * moments[m_i<0>()]) / (static_cast<scalar_t>(6) * moments[m_i<0>()]);  // myy
        moments[m_i<8>()] = -(-static_cast<scalar_t>(6) * myz_I * rho_I + moments[m_i<2>()] * moments[m_i<0>()]) / (static_cast<scalar_t>(3) * moments[m_i<0>()]);                                                                                                                                                                                  // myz
        // moments[m_i<9>()] = moments[m_i<3>()] * moments[m_i<3>()];                                                                                                                                                                                                                                       // mzz
    }
    else
    {
        const scalar_t mxx_I = (pop[q_i<1>()] + pop[q_i<2>()] + pop[q_i<7>()] + pop[q_i<8>()] + pop[q_i<9>()] + pop[q_i<13>()] + pop[q_i<14>()] + pop[q_i<16>()] + pop[q_i<19>()] + pop[q_i<22>()] + pop[q_i<23>()] + pop[q_i<25>()]) * inv_rho_I - velocitySet::cs2<scalar_t>();
        const scalar_t mxy_I = (pop[q_i<7>()] + pop[q_i<8>()] - pop[q_i<13>()] - pop[q_i<14>()] + pop[q_i<19>()] + pop[q_i<22>()] - pop[q_i<23>()] - pop[q_i<25>()]) * inv_rho_I;
        const scalar_t mxz_I = (pop[q_i<9>()] - pop[q_i<16>()] + pop[q_i<19>()] - pop[q_i<22>()] + pop[q_i<23>()] - pop[q_i<25>()]) * inv_rho_I;
        const scalar_t myy_I = (pop[q_i<3>()] + pop[q_i<4>()] + pop[q_i<7>()] + pop[q_i<8>()] + pop[q_i<11>()] + pop[q_i<13>()] + pop[q_i<14>()] + pop[q_i<18>()] + pop[q_i<19>()] + pop[q_i<22>()] + pop[q_i<23>()] + pop[q_i<25>()]) * inv_rho_I - velocitySet::cs2<scalar_t>();
        const scalar_t myz_I = (pop[q_i<11>()] - pop[q_i<18>()] + pop[q_i<19>()] - pop[q_i<22>()] - pop[q_i<23>()] + pop[q_i<25>()]) * inv_rho_I;

        // IRBC-Neumann
        moments[m_i<4>()] = -(-static_cast<scalar_t>(6) * mxx_I * rho_I + static_cast<scalar_t>(6) * myy_I * rho_I - static_cast<scalar_t>(5) * moments[m_i<0>()] * moments[m_i<1>()] * moments[m_i<1>()] - static_cast<scalar_t>(5) * moments[m_i<0>()] * moments[m_i<2>()] * moments[m_i<2>()]) / (static_cast<scalar_t>(10) * moments[m_i<0>()]); // mxx
        moments[m_i<5>()] = (static_cast<scalar_t>(6) * mxy_I * rho_I) / (static_cast<scalar_t>(5) * moments[m_i<0>()]);                                                                                                                                                                                                                             // mxy
        moments[m_i<6>()] = -(-static_cast<scalar_t>(6) * mxz_I * rho_I + moments[m_i<1>()] * moments[m_i<0>()]) / (static_cast<scalar_t>(3) * moments[m_i<0>()]);                                                                                                                                                                                   // mxz
        moments[m_i<7>()] = -(static_cast<scalar_t>(6) * mxx_I * rho_I - static_cast<scalar_t>(6) * myy_I * rho_I - static_cast<scalar_t>(5) * moments[m_i<1>()] * moments[m_i<1>()] * moments[m_i<0>()] - static_cast<scalar_t>(5) * moments[m_i<2>()] * moments[m_i<2>()] * moments[m_i<0>()]) / (static_cast<scalar_t>(10) * moments[m_i<0>()]);  // myy
        moments[m_i<8>()] = -(-static_cast<scalar_t>(6) * myz_I * rho_I + moments[m_i<2>()] * moments[m_i<0>()]) / (static_cast<scalar_t>(3) * moments[m_i<0>()]);                                                                                                                                                                                   // myz
        // moments[m_i<9>()] = moments[m_i<3>()] * moments[m_i<3>()];                                                                                                                                                                                                                                        // mzz
    }

    return;
}

// Edges
case normalVector::WEST_FRONT():
{
    // Classic Neumann
    // const scalar_t moments[m_i<0>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + m_i<0>()];
    // moments[m_i<0>()] = moments[m_i<0>()];
    // moments[m_i<1>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + m_i<1>()];
    // moments[m_i<2>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + m_i<2>()];
    // moments[m_i<3>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + m_i<3>()];

    if constexpr (VelocitySet::Q() == 19)
    {
        const scalar_t mxy_I = (pop[q_i<8>()] - pop[q_i<14>()]) * inv_rho_I;
        const scalar_t myz_I = (pop[q_i<11>()] - pop[q_i<18>()]) * inv_rho_I;

        // IRBC-Neumann
        // moments[m_i<4>()] = moments[m_i<1>()] * moments[m_i<1>()];                                                                   // mxx
        moments[m_i<5>()] = (static_cast<scalar_t>(6) * mxy_I * rho_I + moments[m_i<2>()] * moments[m_i<0>()]) / (static_cast<scalar_t>(3) * moments[m_i<0>()]); // mxy
                                                                                                                                                                 // moments[m_i<6>()] = moments[m_i<1>()] * moments[m_i<3>()];                                                                   // mxz
                                                                                                                                                                 // moments[m_i<7>()] = moments[m_i<2>()] * moments[m_i<2>()];                                                                   // myy
        moments[m_i<8>()] = (static_cast<scalar_t>(6) * myz_I * rho_I - moments[m_i<2>()] * moments[m_i<0>()]) / (static_cast<scalar_t>(3) * moments[m_i<0>()]); // myz
        // moments[m_i<9>()] = moments[m_i<3>()] * moments[m_i<3>()];
    }
    else
    {
        const scalar_t mxy_I = (pop[q_i<8>()] - pop[q_i<14>()] + pop[q_i<22>()] - pop[q_i<25>()]) * inv_rho_I;
        const scalar_t myz_I = (pop[q_i<11>()] - pop[q_i<18>()] - pop[q_i<22>()] + pop[q_i<25>()]) * inv_rho_I;

        // IRBC-Neumann
        // moments[m_i<4>()] = moments[m_i<1>()] * moments[m_i<1>()];                                                                                                                                             // mxx
        moments[m_i<5>()] = -(-static_cast<scalar_t>(45) * mxy_I * rho_I - static_cast<scalar_t>(9) * myz_I * rho_I - static_cast<scalar_t>(5) * moments[m_i<2>()] * moments[m_i<0>()]) / (static_cast<scalar_t>(18) * moments[m_i<0>()]); // mxy
                                                                                                                                                                                                                                           // moments[m_i<6>()] = moments[m_i<1>()] * moments[m_i<3>()];                                                                                                                                             // mxz
                                                                                                                                                                                                                                           // moments[m_i<7>()] = moments[m_i<2>()] * moments[m_i<2>()];                                                                                                                                             // myy
        moments[m_i<8>()] = -(-static_cast<scalar_t>(9) * mxy_I * rho_I - static_cast<scalar_t>(45) * myz_I * rho_I + static_cast<scalar_t>(5) * moments[m_i<2>()] * moments[m_i<0>()]) / (static_cast<scalar_t>(18) * moments[m_i<0>()]); // myz
        // moments[m_i<9>()] = moments[m_i<3>()] * moments[m_i<3>()];                                                                                                                                             // mzz
    }

    return;
}
case normalVector::EAST_FRONT():
{
    // Classic Neumann
    // const scalar_t moments[m_i<0>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + m_i<0>()];
    // moments[m_i<0>()] = moments[m_i<0>()];
    // moments[m_i<1>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + m_i<1>()];
    // moments[m_i<2>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + m_i<2>()];
    // moments[m_i<3>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + m_i<3>()];

    if constexpr (VelocitySet::Q() == 19)
    {
        const scalar_t mxy_I = (pop[q_i<7>()] - pop[q_i<13>()]) * inv_rho_I;
        const scalar_t myz_I = (pop[q_i<11>()] - pop[q_i<18>()]) * inv_rho_I;

        // IRBC-Neumann
        // moments[m_i<4>()] = moments[m_i<1>()] * moments[m_i<1>()];                                                                   // mxx
        moments[m_i<5>()] = (static_cast<scalar_t>(6) * mxy_I * rho_I - moments[m_i<2>()] * moments[m_i<0>()]) / (static_cast<scalar_t>(3) * moments[m_i<0>()]); // mxy
                                                                                                                                                                 // moments[m_i<6>()] = moments[m_i<1>()] * moments[m_i<3>()];                                                                   // mxz
                                                                                                                                                                 // moments[m_i<7>()] = moments[m_i<2>()] * moments[m_i<2>()];                                                                   // myy
        moments[m_i<8>()] = (static_cast<scalar_t>(6) * myz_I * rho_I - moments[m_i<2>()] * moments[m_i<0>()]) / (static_cast<scalar_t>(3) * moments[m_i<0>()]); // myz
        // moments[m_i<9>()] = moments[m_i<3>()] * moments[m_i<3>()];                                                                   // mzz
    }
    else
    {
        const scalar_t mxy_I = (pop[q_i<7>()] - pop[q_i<13>()] + pop[q_i<19>()] - pop[q_i<23>()]) * inv_rho_I;
        const scalar_t myz_I = (pop[q_i<11>()] - pop[q_i<18>()] + pop[q_i<19>()] - pop[q_i<23>()]) * inv_rho_I;

        // IRBC-Neumann
        // moments[m_i<4>()] = moments[m_i<1>()] * moments[m_i<1>()];                                                                                                                                             // mxx
        moments[m_i<5>()] = -(-static_cast<scalar_t>(45) * mxy_I * rho_I + static_cast<scalar_t>(9) * myz_I * rho_I + static_cast<scalar_t>(5) * moments[m_i<2>()] * moments[m_i<0>()]) / (static_cast<scalar_t>(18) * moments[m_i<0>()]); // mxy
                                                                                                                                                                                                                                           // moments[m_i<6>()] = moments[m_i<1>()] * moments[m_i<3>()];                                                                                                                                             // mxz
                                                                                                                                                                                                                                           // moments[m_i<7>()] = moments[m_i<2>()] * moments[m_i<2>()];                                                                                                                                             // myy
        moments[m_i<8>()] = -(static_cast<scalar_t>(9) * mxy_I * rho_I - static_cast<scalar_t>(45) * myz_I * rho_I + static_cast<scalar_t>(5) * moments[m_i<2>()] * moments[m_i<0>()]) / (static_cast<scalar_t>(18) * moments[m_i<0>()]);  // myz
        // moments[m_i<9>()] = moments[m_i<3>()] * moments[m_i<3>()];                                                                                                                                             // mzz
    }

    return;
}
case normalVector::SOUTH_FRONT():
{
    // Classic Neumann
    // const scalar_t moments[m_i<0>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + m_i<0>()];
    // moments[m_i<0>()] = moments[m_i<0>()];
    // moments[m_i<1>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + m_i<1>()];
    // moments[m_i<2>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + m_i<2>()];
    // moments[m_i<3>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + m_i<3>()];

    if constexpr (VelocitySet::Q() == 19)
    {
        const scalar_t mxy_I = (pop[q_i<8>()] - pop[q_i<13>()]) * inv_rho_I;
        const scalar_t mxz_I = (pop[q_i<9>()] - pop[q_i<16>()]) * inv_rho_I;

        // IRBC-Neumann
        // moments[m_i<4>()] = moments[m_i<1>()] * moments[m_i<1>()];                                                                   // mxx
        moments[m_i<5>()] = (static_cast<scalar_t>(6) * mxy_I * rho_I + moments[m_i<1>()] * moments[m_i<0>()]) / (static_cast<scalar_t>(3) * moments[m_i<0>()]); // mxy
        moments[m_i<6>()] = (static_cast<scalar_t>(6) * mxz_I * rho_I - moments[m_i<1>()] * moments[m_i<0>()]) / (static_cast<scalar_t>(3) * moments[m_i<0>()]); // mxz
        // moments[m_i<7>()] = moments[m_i<2>()] * moments[m_i<2>()];                                                                   // myy
        // moments[m_i<8>()] = moments[m_i<2>()] * moments[m_i<3>()];                                                                   // myz
        // moments[m_i<9>()] = moments[m_i<3>()] * moments[m_i<3>()];                                                                   // mzz
    }
    else
    {
        const scalar_t mxy_I = (pop[q_i<8>()] - pop[q_i<13>()] + pop[q_i<22>()] - pop[q_i<23>()]) * inv_rho_I;
        const scalar_t mxz_I = (pop[q_i<9>()] - pop[q_i<16>()] - pop[q_i<22>()] + pop[q_i<23>()]) * inv_rho_I;

        // IRBC-Neumann
        // moments[m_i<4>()] = moments[m_i<1>()] * moments[m_i<1>()];                                                                                                                                             // mxx
        moments[m_i<5>()] = -(-static_cast<scalar_t>(45) * mxy_I * rho_I - static_cast<scalar_t>(9) * mxz_I * rho_I - static_cast<scalar_t>(5) * moments[m_i<1>()] * moments[m_i<0>()]) / (static_cast<scalar_t>(18) * moments[m_i<0>()]); // mxy
        moments[m_i<6>()] = -(-static_cast<scalar_t>(9) * mxy_I * rho_I - static_cast<scalar_t>(45) * mxz_I * rho_I + static_cast<scalar_t>(5) * moments[m_i<1>()] * moments[m_i<0>()]) / (static_cast<scalar_t>(18) * moments[m_i<0>()]); // mxz
        // moments[m_i<7>()] = moments[m_i<2>()] * moments[m_i<2>()];                                                                                                                                             // myy
        // moments[m_i<8>()] = moments[m_i<2>()] * moments[m_i<3>()];                                                                                                                                             // myz
        // moments[m_i<9>()] = moments[m_i<3>()] * moments[m_i<3>()];                                                                                                                                             // mzz
    }

    return;
}
case normalVector::NORTH_FRONT():
{
    // Classic Neumann
    // const scalar_t moments[m_i<0>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + m_i<0>()];
    // moments[m_i<0>()] = moments[m_i<0>()];
    // moments[m_i<1>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + m_i<1>()];
    // moments[m_i<2>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + m_i<2>()];
    // moments[m_i<3>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + m_i<3>()];

    if constexpr (VelocitySet::Q() == 19)
    {
        const scalar_t mxy_I = (pop[q_i<7>()] - pop[q_i<14>()]) * inv_rho_I;
        const scalar_t mxz_I = (pop[q_i<9>()] - pop[q_i<16>()]) * inv_rho_I;

        // IRBC-Neumann
        // moments[m_i<4>()] = moments[m_i<1>()] * moments[m_i<1>()];                                                                   // mxx
        moments[m_i<5>()] = (static_cast<scalar_t>(6) * mxy_I * rho_I - moments[m_i<1>()] * moments[m_i<0>()]) / (static_cast<scalar_t>(3) * moments[m_i<0>()]); // mxy
        moments[m_i<6>()] = (static_cast<scalar_t>(6) * mxz_I * rho_I - moments[m_i<1>()] * moments[m_i<0>()]) / (static_cast<scalar_t>(3) * moments[m_i<0>()]); // mxz
        // moments[m_i<7>()] = moments[m_i<2>()] * moments[m_i<2>()];                                                                   // myy
        // moments[m_i<8>()] = moments[m_i<2>()] * moments[m_i<3>()];                                                                   // myz
        // moments[m_i<9>()] = moments[m_i<3>()] * moments[m_i<3>()];                                                                   // mzz
    }
    else
    {
        const scalar_t mxy_I = (pop[q_i<7>()] - pop[q_i<14>()] + pop[q_i<19>()] - pop[q_i<25>()]) * inv_rho_I;
        const scalar_t mxz_I = (pop[q_i<9>()] - pop[q_i<16>()] + pop[q_i<19>()] - pop[q_i<25>()]) * inv_rho_I;

        // IRBC-Neumann
        // moments[m_i<4>()] = moments[m_i<1>()] * moments[m_i<1>()];                                                                                                                                             // mxx
        moments[m_i<5>()] = -(-static_cast<scalar_t>(45) * mxy_I * rho_I + static_cast<scalar_t>(9) * mxz_I * rho_I + static_cast<scalar_t>(5) * moments[m_i<1>()] * moments[m_i<0>()]) / (static_cast<scalar_t>(18) * moments[m_i<0>()]); // mxy
        moments[m_i<6>()] = -(static_cast<scalar_t>(9) * mxy_I * rho_I - static_cast<scalar_t>(45) * mxz_I * rho_I + static_cast<scalar_t>(5) * moments[m_i<1>()] * moments[m_i<0>()]) / (static_cast<scalar_t>(18) * moments[m_i<0>()]);  // mxz
        // moments[m_i<7>()] = moments[m_i<2>()] * moments[m_i<2>()];                                                                                                                                             // myy
        // moments[m_i<8>()] = moments[m_i<2>()] * moments[m_i<3>()];                                                                                                                                             // myz
        // moments[m_i<9>()] = moments[m_i<3>()] * moments[m_i<3>()];                                                                                                                                             // mzz
    }

    return;
}

// Edges
case normalVector::SOUTH_WEST_FRONT():
{
    // Classic Neumann
    // moments[m_i<0>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + m_i<0>()];
    // moments[m_i<0>()] = rho0();
    // moments[m_i<1>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + m_i<1>()];
    // moments[m_i<2>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + m_i<2>()];
    // moments[m_i<3>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + m_i<3>()];

    // IRBC-Neumann
    // moments[m_i<4>()] = moments[m_i<1>()] * moments[m_i<1>()]; // mxx
    // moments[m_i<5>()] = moments[m_i<1>()] * moments[m_i<2>()]; // mxy
    // moments[m_i<6>()] = moments[m_i<1>()] * moments[m_i<3>()]; // mxz
    // moments[m_i<7>()] = moments[m_i<2>()] * moments[m_i<2>()]; // myy
    // moments[m_i<8>()] = moments[m_i<2>()] * moments[m_i<3>()]; // myz
    // moments[m_i<9>()] = moments[m_i<3>()] * moments[m_i<3>()]; // mzz

    return;
}
case normalVector::NORTH_WEST_FRONT():
{
    // Classic Neumann
    // moments[m_i<0>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + m_i<0>()];
    // moments[m_i<0>()] = rho0();
    // moments[m_i<1>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + m_i<1>()];
    // moments[m_i<2>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + m_i<2>()];
    // moments[m_i<3>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + m_i<3>()];

    // IRBC-Neumann
    // moments[m_i<4>()] = moments[m_i<1>()] * moments[m_i<1>()]; // mxx
    // moments[m_i<5>()] = moments[m_i<1>()] * moments[m_i<2>()]; // mxy
    // moments[m_i<6>()] = moments[m_i<1>()] * moments[m_i<3>()]; // mxz
    // moments[m_i<7>()] = moments[m_i<2>()] * moments[m_i<2>()]; // myy
    // moments[m_i<8>()] = moments[m_i<2>()] * moments[m_i<3>()]; // myz
    // moments[m_i<9>()] = moments[m_i<3>()] * moments[m_i<3>()]; // mzz

    return;
}
case normalVector::SOUTH_EAST_FRONT():
{
    // Classic Neumann
    // moments[m_i<0>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + m_i<0>()];
    // moments[m_i<0>()] = rho0();
    // moments[m_i<1>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + m_i<1>()];
    // moments[m_i<2>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + m_i<2>()];
    // moments[m_i<3>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + m_i<3>()];

    // IRBC-Neumann
    // moments[m_i<4>()] = moments[m_i<1>()] * moments[m_i<1>()]; // mxx
    // moments[m_i<5>()] = moments[m_i<1>()] * moments[m_i<2>()]; // mxy
    // moments[m_i<6>()] = moments[m_i<1>()] * moments[m_i<3>()]; // mxz
    // moments[m_i<7>()] = moments[m_i<2>()] * moments[m_i<2>()]; // myy
    // moments[m_i<8>()] = moments[m_i<2>()] * moments[m_i<3>()]; // myz
    // moments[m_i<9>()] = moments[m_i<3>()] * moments[m_i<3>()]; // mzz

    return;
}
case normalVector::NORTH_EAST_FRONT():
{
    // Classic Neumann
    // moments[m_i<0>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + m_i<0>()];
    // moments[m_i<0>()] = rho0();
    // moments[m_i<1>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + m_i<1>()];
    // moments[m_i<2>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + m_i<2>()];
    // moments[m_i<3>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + m_i<3>()];

    // IRBC-Neumann
    // moments[m_i<4>()] = moments[m_i<1>()] * moments[m_i<1>()]; // mxx
    // moments[m_i<5>()] = moments[m_i<1>()] * moments[m_i<2>()]; // mxy
    // moments[m_i<6>()] = moments[m_i<1>()] * moments[m_i<3>()]; // mxz
    // moments[m_i<7>()] = moments[m_i<2>()] * moments[m_i<2>()]; // myy
    // moments[m_i<8>()] = moments[m_i<2>()] * moments[m_i<3>()]; // myz
    // moments[m_i<9>()] = moments[m_i<3>()] * moments[m_i<3>()]; // mzz

    return;
}
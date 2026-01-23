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
    Applies static boundary conditions

SourceFiles
    static.cuh

Notes
    This file is intended to be included directly inside a switch-case block.
    Do NOT use include guards (#ifndef/#define/#endif).

\*---------------------------------------------------------------------------*/

// Static corners
case normalVector::WEST_SOUTH_BACK():
{
    if constexpr (VelocitySet::Q() == 19)
    {
        moments[m_i<0>()] = static_cast<scalar_t>(12) * p_I / static_cast<scalar_t>(7); // p
    }
    else
    {
        moments[m_i<0>()] = static_cast<scalar_t>(216) * p_I / static_cast<scalar_t>(125); // p
    }

    moments[m_i<1>()] = static_cast<scalar_t>(0);  // ux
    moments[m_i<2>()] = static_cast<scalar_t>(0);  // uy
    moments[m_i<3>()] = static_cast<scalar_t>(0);  // uz
    moments[m_i<4>()] = static_cast<scalar_t>(0);  // mxx
    moments[m_i<5>()] = static_cast<scalar_t>(0);  // mxy
    moments[m_i<6>()] = static_cast<scalar_t>(0);  // mxz
    moments[m_i<7>()] = static_cast<scalar_t>(0);  // myy
    moments[m_i<8>()] = static_cast<scalar_t>(0);  // myz
    moments[m_i<9>()] = static_cast<scalar_t>(0);  // mzz
    moments[m_i<10>()] = static_cast<scalar_t>(0); // phi

    return;
}
case normalVector::WEST_SOUTH_FRONT():
{
    if constexpr (VelocitySet::Q() == 19)
    {
        moments[m_i<0>()] = static_cast<scalar_t>(12) * p_I / static_cast<scalar_t>(7); // p
    }
    else
    {
        moments[m_i<0>()] = static_cast<scalar_t>(216) * p_I / static_cast<scalar_t>(125); // p
    }

    moments[m_i<1>()] = static_cast<scalar_t>(0);  // ux
    moments[m_i<2>()] = static_cast<scalar_t>(0);  // uy
    moments[m_i<3>()] = static_cast<scalar_t>(0);  // uz
    moments[m_i<4>()] = static_cast<scalar_t>(0);  // mxx
    moments[m_i<5>()] = static_cast<scalar_t>(0);  // mxy
    moments[m_i<6>()] = static_cast<scalar_t>(0);  // mxz
    moments[m_i<7>()] = static_cast<scalar_t>(0);  // myy
    moments[m_i<8>()] = static_cast<scalar_t>(0);  // myz
    moments[m_i<9>()] = static_cast<scalar_t>(0);  // mzz
    moments[m_i<10>()] = static_cast<scalar_t>(0); // phi

    return;
}
case normalVector::EAST_SOUTH_BACK():
{
    if constexpr (VelocitySet::Q() == 19)
    {
        moments[m_i<0>()] = static_cast<scalar_t>(12) * p_I / static_cast<scalar_t>(7); // p
    }
    else
    {
        moments[m_i<0>()] = static_cast<scalar_t>(216) * p_I / static_cast<scalar_t>(125); // p
    }

    moments[m_i<1>()] = static_cast<scalar_t>(0);  // ux
    moments[m_i<2>()] = static_cast<scalar_t>(0);  // uy
    moments[m_i<3>()] = static_cast<scalar_t>(0);  // uz
    moments[m_i<4>()] = static_cast<scalar_t>(0);  // mxx
    moments[m_i<5>()] = static_cast<scalar_t>(0);  // mxy
    moments[m_i<6>()] = static_cast<scalar_t>(0);  // mxz
    moments[m_i<7>()] = static_cast<scalar_t>(0);  // myy
    moments[m_i<8>()] = static_cast<scalar_t>(0);  // myz
    moments[m_i<9>()] = static_cast<scalar_t>(0);  // mzz
    moments[m_i<10>()] = static_cast<scalar_t>(0); // phi

    return;
}
case normalVector::EAST_SOUTH_FRONT():
{
    if constexpr (VelocitySet::Q() == 19)
    {
        moments[m_i<0>()] = static_cast<scalar_t>(12) * p_I / static_cast<scalar_t>(7); // p
    }
    else
    {
        moments[m_i<0>()] = static_cast<scalar_t>(216) * p_I / static_cast<scalar_t>(125); // p
    }

    moments[m_i<1>()] = static_cast<scalar_t>(0);  // ux
    moments[m_i<2>()] = static_cast<scalar_t>(0);  // uy
    moments[m_i<3>()] = static_cast<scalar_t>(0);  // uz
    moments[m_i<4>()] = static_cast<scalar_t>(0);  // mxx
    moments[m_i<5>()] = static_cast<scalar_t>(0);  // mxy
    moments[m_i<6>()] = static_cast<scalar_t>(0);  // mxz
    moments[m_i<7>()] = static_cast<scalar_t>(0);  // myy
    moments[m_i<8>()] = static_cast<scalar_t>(0);  // myz
    moments[m_i<9>()] = static_cast<scalar_t>(0);  // mzz
    moments[m_i<10>()] = static_cast<scalar_t>(0); // phi

    return;
}
case normalVector::WEST_NORTH_BACK():
{
    if constexpr (VelocitySet::Q() == 19)
    {
        moments[m_i<0>()] = static_cast<scalar_t>(12) * p_I / static_cast<scalar_t>(7); // p
    }
    else
    {
        moments[m_i<0>()] = static_cast<scalar_t>(216) * p_I / static_cast<scalar_t>(125); // p
    }

    moments[m_i<1>()] = static_cast<scalar_t>(0);  // ux
    moments[m_i<2>()] = static_cast<scalar_t>(0);  // uy
    moments[m_i<3>()] = static_cast<scalar_t>(0);  // uz
    moments[m_i<4>()] = static_cast<scalar_t>(0);  // mxx
    moments[m_i<5>()] = static_cast<scalar_t>(0);  // mxy
    moments[m_i<6>()] = static_cast<scalar_t>(0);  // mxz
    moments[m_i<7>()] = static_cast<scalar_t>(0);  // myy
    moments[m_i<8>()] = static_cast<scalar_t>(0);  // myz
    moments[m_i<9>()] = static_cast<scalar_t>(0);  // mzz
    moments[m_i<10>()] = static_cast<scalar_t>(0); // phi

    return;
}
// case normalVector::WEST_NORTH_FRONT():
// {
//     if constexpr (VelocitySet::Q() == 19)
//     {
//         moments[m_i<0>()] = static_cast<scalar_t>(12) * p_I / static_cast<scalar_t>(7); // p
//     }
//     else
//     {
//         moments[m_i<0>()] = static_cast<scalar_t>(216) * p_I / static_cast<scalar_t>(125); // p
//     }

//     moments[m_i<1>()] = static_cast<scalar_t>(0);  // ux
//     moments[m_i<2>()] = static_cast<scalar_t>(0);  // uy
//     moments[m_i<3>()] = static_cast<scalar_t>(0);  // uz
//     moments[m_i<4>()] = static_cast<scalar_t>(0);  // mxx
//     moments[m_i<5>()] = static_cast<scalar_t>(0);  // mxy
//     moments[m_i<6>()] = static_cast<scalar_t>(0);  // mxz
//     moments[m_i<7>()] = static_cast<scalar_t>(0);  // myy
//     moments[m_i<8>()] = static_cast<scalar_t>(0);  // myz
//     moments[m_i<9>()] = static_cast<scalar_t>(0);  // mzz
//     moments[m_i<10>()] = static_cast<scalar_t>(0); // phi

//     return;
// }
case normalVector::EAST_NORTH_BACK():
{
    if constexpr (VelocitySet::Q() == 19)
    {
        moments[m_i<0>()] = static_cast<scalar_t>(12) * p_I / static_cast<scalar_t>(7); // p
    }
    else
    {
        moments[m_i<0>()] = static_cast<scalar_t>(216) * p_I / static_cast<scalar_t>(125); // p
    }

    moments[m_i<1>()] = static_cast<scalar_t>(0);  // ux
    moments[m_i<2>()] = static_cast<scalar_t>(0);  // uy
    moments[m_i<3>()] = static_cast<scalar_t>(0);  // uz
    moments[m_i<4>()] = static_cast<scalar_t>(0);  // mxx
    moments[m_i<5>()] = static_cast<scalar_t>(0);  // mxy
    moments[m_i<6>()] = static_cast<scalar_t>(0);  // mxz
    moments[m_i<7>()] = static_cast<scalar_t>(0);  // myy
    moments[m_i<8>()] = static_cast<scalar_t>(0);  // myz
    moments[m_i<9>()] = static_cast<scalar_t>(0);  // mzz
    moments[m_i<10>()] = static_cast<scalar_t>(0); // phi

    return;
}
// case normalVector::EAST_NORTH_FRONT():
// {
//     if constexpr (VelocitySet::Q() == 19)
//     {
//         moments[m_i<0>()] = static_cast<scalar_t>(12) * p_I / static_cast<scalar_t>(7); // p
//     }
//     else
//     {
//         moments[m_i<0>()] = static_cast<scalar_t>(216) * p_I / static_cast<scalar_t>(125); // p
//     }

//     moments[m_i<1>()] = static_cast<scalar_t>(0);  // ux
//     moments[m_i<2>()] = static_cast<scalar_t>(0);  // uy
//     moments[m_i<3>()] = static_cast<scalar_t>(0);  // uz
//     moments[m_i<4>()] = static_cast<scalar_t>(0);  // mxx
//     moments[m_i<5>()] = static_cast<scalar_t>(0);  // mxy
//     moments[m_i<6>()] = static_cast<scalar_t>(0);  // mxz
//     moments[m_i<7>()] = static_cast<scalar_t>(0);  // myy
//     moments[m_i<8>()] = static_cast<scalar_t>(0);  // myz
//     moments[m_i<9>()] = static_cast<scalar_t>(0);  // mzz
//     moments[m_i<10>()] = static_cast<scalar_t>(0); // phi

//     return;
// }

// Static edges
case normalVector::WEST_SOUTH():
{
    const scalar_t mxy_I = WEST_SOUTH_mxy_I(pop);

    const scalar_t p = static_cast<scalar_t>(36) * (-mxy_I + p_I + mxy_I * device::omega) / (static_cast<scalar_t>(24) + device::omega);
    const scalar_t mxy = static_cast<scalar_t>(4) * (static_cast<scalar_t>(25) * mxy_I - p_I) / (static_cast<scalar_t>(24) + device::omega);

    moments[m_i<0>()] = p;                         // p
    moments[m_i<1>()] = static_cast<scalar_t>(0);  // ux
    moments[m_i<2>()] = static_cast<scalar_t>(0);  // uy
    moments[m_i<3>()] = static_cast<scalar_t>(0);  // uz
    moments[m_i<4>()] = static_cast<scalar_t>(0);  // mxx
    moments[m_i<5>()] = mxy;                       // mxy
    moments[m_i<6>()] = static_cast<scalar_t>(0);  // mxz
    moments[m_i<7>()] = static_cast<scalar_t>(0);  // myy
    moments[m_i<8>()] = static_cast<scalar_t>(0);  // myz
    moments[m_i<9>()] = static_cast<scalar_t>(0);  // mzz
    moments[m_i<10>()] = static_cast<scalar_t>(0); // phi

    return;
}
case normalVector::EAST_SOUTH():
{
    const scalar_t mxy_I = EAST_SOUTH_mxy_I(pop);

    const scalar_t p = -static_cast<scalar_t>(36) * (-mxy_I - p_I + mxy_I * device::omega) / (static_cast<scalar_t>(24) + device::omega);
    const scalar_t mxy = static_cast<scalar_t>(4) * (static_cast<scalar_t>(25) * mxy_I + p_I) / (static_cast<scalar_t>(24) + device::omega);

    moments[m_i<0>()] = p;                         // p
    moments[m_i<1>()] = static_cast<scalar_t>(0);  // ux
    moments[m_i<2>()] = static_cast<scalar_t>(0);  // uy
    moments[m_i<3>()] = static_cast<scalar_t>(0);  // uz
    moments[m_i<4>()] = static_cast<scalar_t>(0);  // mxx
    moments[m_i<5>()] = mxy;                       // mxy
    moments[m_i<6>()] = static_cast<scalar_t>(0);  // mxz
    moments[m_i<7>()] = static_cast<scalar_t>(0);  // myy
    moments[m_i<8>()] = static_cast<scalar_t>(0);  // myz
    moments[m_i<9>()] = static_cast<scalar_t>(0);  // mzz
    moments[m_i<10>()] = static_cast<scalar_t>(0); // phi

    return;
}
case normalVector::WEST_BACK():
{
    const scalar_t mxz_I = WEST_BACK_mxz_I(pop);

    const scalar_t p = static_cast<scalar_t>(36) * (-mxz_I + p_I + mxz_I * device::omega) / (static_cast<scalar_t>(24) + device::omega);
    const scalar_t mxz = static_cast<scalar_t>(4) * (static_cast<scalar_t>(25) * mxz_I - p_I) / (static_cast<scalar_t>(24) + device::omega);

    moments[m_i<0>()] = p;                         // p
    moments[m_i<1>()] = static_cast<scalar_t>(0);  // ux
    moments[m_i<2>()] = static_cast<scalar_t>(0);  // uy
    moments[m_i<3>()] = static_cast<scalar_t>(0);  // uz
    moments[m_i<4>()] = static_cast<scalar_t>(0);  // mxx
    moments[m_i<5>()] = static_cast<scalar_t>(0);  // mxy
    moments[m_i<6>()] = mxz;                       // mxz
    moments[m_i<7>()] = static_cast<scalar_t>(0);  // myy
    moments[m_i<8>()] = static_cast<scalar_t>(0);  // myz
    moments[m_i<9>()] = static_cast<scalar_t>(0);  // mzz
    moments[m_i<10>()] = static_cast<scalar_t>(0); // phi

    return;
}
case normalVector::EAST_BACK():
{
    const scalar_t mxz_I = EAST_BACK_mxz_I(pop);

    const scalar_t p = -static_cast<scalar_t>(36) * (-mxz_I - p_I + mxz_I * device::omega) / (static_cast<scalar_t>(24) + device::omega);
    const scalar_t mxz = static_cast<scalar_t>(4) * (static_cast<scalar_t>(25) * mxz_I + p_I) / (static_cast<scalar_t>(24) + device::omega);

    moments[m_i<0>()] = p;                         // p
    moments[m_i<1>()] = static_cast<scalar_t>(0);  // ux
    moments[m_i<2>()] = static_cast<scalar_t>(0);  // uy
    moments[m_i<3>()] = static_cast<scalar_t>(0);  // uz
    moments[m_i<4>()] = static_cast<scalar_t>(0);  // mxx
    moments[m_i<5>()] = static_cast<scalar_t>(0);  // mxy
    moments[m_i<6>()] = mxz;                       // mxz
    moments[m_i<7>()] = static_cast<scalar_t>(0);  // myy
    moments[m_i<8>()] = static_cast<scalar_t>(0);  // myz
    moments[m_i<9>()] = static_cast<scalar_t>(0);  // mzz
    moments[m_i<10>()] = static_cast<scalar_t>(0); // phi

    return;
}
case normalVector::SOUTH_BACK():
{
    const scalar_t myz_I = SOUTH_BACK_myz_I(pop);

    const scalar_t p = static_cast<scalar_t>(36) * (-myz_I + p_I + myz_I * device::omega) / (static_cast<scalar_t>(24) + device::omega);
    const scalar_t myz = static_cast<scalar_t>(4) * (static_cast<scalar_t>(25) * myz_I - p_I) / (static_cast<scalar_t>(24) + device::omega);

    moments[m_i<0>()] = p;                         // p
    moments[m_i<1>()] = static_cast<scalar_t>(0);  // ux
    moments[m_i<2>()] = static_cast<scalar_t>(0);  // uy
    moments[m_i<3>()] = static_cast<scalar_t>(0);  // uz
    moments[m_i<4>()] = static_cast<scalar_t>(0);  // mxx
    moments[m_i<5>()] = static_cast<scalar_t>(0);  // mxy
    moments[m_i<6>()] = static_cast<scalar_t>(0);  // mxz
    moments[m_i<7>()] = static_cast<scalar_t>(0);  // myy
    moments[m_i<8>()] = myz;                       // myz
    moments[m_i<9>()] = static_cast<scalar_t>(0);  // mzz
    moments[m_i<10>()] = static_cast<scalar_t>(0); // phi

    return;
}
case normalVector::SOUTH_FRONT():
{
    const scalar_t myz_I = SOUTH_FRONT_myz_I(pop);

    const scalar_t p = -static_cast<scalar_t>(36) * (-myz_I - p_I + myz_I * device::omega) / (static_cast<scalar_t>(24) + device::omega);
    const scalar_t myz = static_cast<scalar_t>(4) * (static_cast<scalar_t>(25) * myz_I + p_I) / (static_cast<scalar_t>(24) + device::omega);

    moments[m_i<0>()] = p;                         // p
    moments[m_i<1>()] = static_cast<scalar_t>(0);  // ux
    moments[m_i<2>()] = static_cast<scalar_t>(0);  // uy
    moments[m_i<3>()] = static_cast<scalar_t>(0);  // uz
    moments[m_i<4>()] = static_cast<scalar_t>(0);  // mxx
    moments[m_i<5>()] = static_cast<scalar_t>(0);  // mxy
    moments[m_i<6>()] = static_cast<scalar_t>(0);  // mxz
    moments[m_i<7>()] = static_cast<scalar_t>(0);  // myy
    moments[m_i<8>()] = myz;                       // myz
    moments[m_i<9>()] = static_cast<scalar_t>(0);  // mzz
    moments[m_i<10>()] = static_cast<scalar_t>(0); // phi

    return;
}
case normalVector::NORTH_BACK():
{
    const scalar_t myz_I = NORTH_BACK_myz_I(pop);

    const scalar_t p = -static_cast<scalar_t>(36) * (-myz_I - p_I + myz_I * device::omega) / (static_cast<scalar_t>(24) + device::omega);
    const scalar_t myz = static_cast<scalar_t>(4) * (static_cast<scalar_t>(25) * myz_I + p_I) / (static_cast<scalar_t>(24) + device::omega);

    moments[m_i<0>()] = p;                         // p
    moments[m_i<1>()] = static_cast<scalar_t>(0);  // ux
    moments[m_i<2>()] = static_cast<scalar_t>(0);  // uy
    moments[m_i<3>()] = static_cast<scalar_t>(0);  // uz
    moments[m_i<4>()] = static_cast<scalar_t>(0);  // mxx
    moments[m_i<5>()] = static_cast<scalar_t>(0);  // mxy
    moments[m_i<6>()] = static_cast<scalar_t>(0);  // mxz
    moments[m_i<7>()] = static_cast<scalar_t>(0);  // myy
    moments[m_i<8>()] = myz;                       // myz
    moments[m_i<9>()] = static_cast<scalar_t>(0);  // mzz
    moments[m_i<10>()] = static_cast<scalar_t>(0); // phi

    return;
}
// case normalVector::NORTH_FRONT():
// {
//     const scalar_t myz_I = NORTH_FRONT_myz_I(pop);

//     const scalar_t p = static_cast<scalar_t>(36) * (-myz_I + p_I + myz_I * device::omega) / (static_cast<scalar_t>(24) + device::omega);
//     const scalar_t myz = static_cast<scalar_t>(4) * (static_cast<scalar_t>(25) * myz_I - p_I) / (static_cast<scalar_t>(24) + device::omega);

//     moments[m_i<0>()] = p;                         // p
//     moments[m_i<1>()] = static_cast<scalar_t>(0);  // ux
//     moments[m_i<2>()] = static_cast<scalar_t>(0);  // uy
//     moments[m_i<3>()] = static_cast<scalar_t>(0);  // uz
//     moments[m_i<4>()] = static_cast<scalar_t>(0);  // mxx
//     moments[m_i<5>()] = static_cast<scalar_t>(0);  // mxy
//     moments[m_i<6>()] = static_cast<scalar_t>(0);  // mxz
//     moments[m_i<7>()] = static_cast<scalar_t>(0);  // myy
//     moments[m_i<8>()] = myz;                       // myz
//     moments[m_i<9>()] = static_cast<scalar_t>(0);  // mzz
//     moments[m_i<10>()] = static_cast<scalar_t>(0); // phi

//     return;
// }
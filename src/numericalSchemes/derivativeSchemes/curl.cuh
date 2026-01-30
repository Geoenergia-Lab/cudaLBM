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
    Curl of a vector field

Namespace
    LBM

SourceFiles
    curl.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_CURL_CUH
#define __MBLBM_CURL_CUH

namespace LBM
{
    namespace derivative
    {
        /**
         * @brief Calculates the curl of a vector field
         * @return The curl of (u, v, w)
         * @param u The x component of the vector
         * @param v The y component of the vector
         * @param w The z component of the vector
         * @param mesh The lattice mesh
         **/
        template <const label_t order_, typename T, class LatticeMesh>
        __host__ [[nodiscard]] const std::vector<std::vector<T>> curl(
            const std::vector<T> &u,
            const std::vector<T> &v,
            const std::vector<T> &w,
            const LatticeMesh &mesh)
        {
            // Calculate the derivatives
            const std::vector<double> dwdy = dfdy<order_, double>(w, mesh);
            const std::vector<double> dvdz = dfdz<order_, double>(v, mesh);

            const std::vector<double> dudz = dfdz<order_, double>(u, mesh);
            const std::vector<double> dwdx = dfdx<order_, double>(w, mesh);

            const std::vector<double> dvdx = dfdx<order_, double>(v, mesh);
            const std::vector<double> dudy = dfdy<order_, double>(u, mesh);

            std::vector<T> curl_x(u.size(), 0);
            std::vector<T> curl_y(u.size(), 0);
            std::vector<T> curl_z(u.size(), 0);

            for (label_t i = 0; i < curl_x.size(); i++)
            {
                curl_x[i] = static_cast<T>(dwdy[i] - dvdz[i]);
                curl_y[i] = static_cast<T>(dudz[i] - dwdx[i]);
                curl_z[i] = static_cast<T>(dvdx[i] - dudy[i]);
            }

            return {curl_x, curl_y, curl_z};
        }
    }
}

#endif
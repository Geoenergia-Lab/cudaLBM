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
    Divergence of a vector field

Namespace
    LBM

SourceFiles
    div.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_DIV_CUH
#define __MBLBM_DIV_CUH

namespace LBM
{
    namespace numericalSchemes
    {
        namespace derivative
        {
            /**
             * @brief Calculates the divergence of a vector field
             * @return The divergence of (u, v, w)
             * @param[in] u The x component of the vector
             * @param[in] v The y component of the vector
             * @param[in] w The z component of the vector
             * @param[in] mesh The lattice mesh
             **/
            template <const label_t SchemeOrder, typename T>
            __host__ [[nodiscard]] const std::vector<T> div(
                const std::vector<T> &u,
                const std::vector<T> &v,
                const std::vector<T> &w,
                const host::latticeMesh &mesh)
            {
                // Calculate the components of div
                const std::vector<double> dudx = dfdx<SchemeOrder, double>(u, mesh);
                const std::vector<double> dvdy = dfdy<SchemeOrder, double>(v, mesh);
                const std::vector<double> dwdz = dfdz<SchemeOrder, double>(w, mesh);

                // Sum the components
                std::vector<T> divu(dudx.size(), 0);
                for (label_t i = 0; i < dudx.size(); i++)
                {
                    divu[i] = static_cast<T>(dudx[i] + dvdy[i] + dwdz[i]);
                }

                // Return div
                return divu;
            }
        }
    }
}

#endif
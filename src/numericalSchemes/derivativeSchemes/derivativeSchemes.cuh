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
    Numerical differentiation schemes

Namespace
    LBM

SourceFiles
    derivativeSchemes.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_DERIVATIVESCHEMES_CUH
#define __MBLBM_DERIVATIVESCHEMES_CUH

namespace LBM
{
    namespace derivative
    {
        /**
         * @brief Calculates the x derivative of a scalar field
         * @return The x-derivative of f
         * @param f The field to be differentiated
         * @param mesh The lattice mesh
         **/
        template <const label_t order_, typename TReturn, typename T, class LatticeMesh>
        __host__ [[nodiscard]] const std::vector<TReturn> dfdx(
            const std::vector<T> &f,
            const LatticeMesh &mesh)
        {
            static_assert((order_ == 2) | (order_ == 4) | (order_ == 6) | (order_ == 8), "Invalid finite difference scheme order: valid orders are 2, 4, 6 and 8");

#ifdef MULTI_GPU

            static_assert(false, "derivative::dfdx not implemented for multi GPU yet");

#else

            const label_t nx = mesh.nx();
            const label_t ny = mesh.ny();
            const label_t nz = mesh.nz();

            const double dx = 1;

            std::vector<TReturn> dfdx(f.size(), 0);
            constexpr const label_t pad = order_ - 1;
            const label_t nx_padded = nx + 2 * pad;
            std::vector<double> padded_line(nx_padded, 0);

            for (label_t z = 0; z < nz; ++z)
            {
                for (label_t y = 0; y < ny; ++y)
                {
                    // Fill interior region of padded_line
                    for (label_t x = 0; x < nx; ++x)
                    {
                        padded_line[pad + x] = static_cast<double>(f[host::idxScalarGlobal(x, y, z, nx, ny)]);
                    }

                    // Set left ghost cells (reflect & negate)
                    for (label_t i = 0; i < pad; ++i)
                    {
                        padded_line[i] = static_cast<double>(-f[host::idxScalarGlobal(pad - i, y, z, nx, ny)]);
                    }

                    // Set right ghost cells (reflect & negate)
                    for (label_t i = 0; i < pad; ++i)
                    {
                        padded_line[pad + nx + i] = static_cast<double>(-f[host::idxScalarGlobal(nx - 2 - i, y, z, nx, ny)]);
                    }

                    // Compute derivatives for each point in x-direction
                    for (label_t x = 0; x < nx; ++x)
                    {
                        const label_t center = pad + x;

                        if constexpr (order_ == 2)
                        {
                            dfdx[host::idxScalarGlobal(x, y, z, nx, ny)] = static_cast<TReturn>(
                                (padded_line[center + 1] - padded_line[center - 1]) / (2.0 * static_cast<double>(dx)));
                        }

                        if constexpr (order_ == 4)
                        {
                            dfdx[host::idxScalarGlobal(x, y, z, nx, ny)] = static_cast<TReturn>(
                                (2.0 / 3.0 * (padded_line[center + 1] - padded_line[center - 1]) +
                                 1.0 / 12.0 * (padded_line[center + 2] - padded_line[center - 2])) /
                                static_cast<double>(dx));
                        }

                        if constexpr (order_ == 6)
                        {
                            dfdx[host::idxScalarGlobal(x, y, z, nx, ny)] = static_cast<TReturn>(
                                (3.0 / 4.0 * (padded_line[center + 1] - padded_line[center - 1]) +
                                 3.0 / 20.0 * (padded_line[center + 2] - padded_line[center - 2]) +
                                 1.0 / 60.0 * (padded_line[center + 3] - padded_line[center - 3])) /
                                static_cast<double>(dx));
                        }

                        if constexpr (order_ == 8)
                        {
                            dfdx[host::idxScalarGlobal(x, y, z, nx, ny)] = static_cast<TReturn>(
                                (4.0 / 5.0 * (padded_line[center + 1] - padded_line[center - 1]) +
                                 1.0 / 5.0 * (padded_line[center + 2] - padded_line[center - 2]) +
                                 4.0 / 105.0 * (padded_line[center + 3] - padded_line[center - 3]) +
                                 1.0 / 280.0 * (padded_line[center + 4] - padded_line[center - 4])) /
                                static_cast<double>(dx));
                        }
                    }
                }
            }
            return dfdx;

#endif
        }

        /**
         * @brief Calculates the y derivative of a scalar field
         * @return The y-derivative of f
         * @param f The field to be differentiated
         * @param mesh The lattice mesh
         **/
        template <const label_t order_, typename TReturn, typename T, class LatticeMesh>
        __host__ [[nodiscard]] const std::vector<TReturn> dfdy(
            const std::vector<T> &f,
            const LatticeMesh &mesh)
        {
            static_assert((order_ == 2) | (order_ == 4) | (order_ == 6) | (order_ == 8), "Invalid finite difference scheme order: valid orders are 2, 4, 6 and 8");

#ifdef MULTI_GPU

            static_assert(false, "derivative::dfdy not implemented for multi GPU yet");

#else

            const label_t nx = mesh.nx();
            const label_t ny = mesh.ny();
            const label_t nz = mesh.nz();
            constexpr const double dy = 1; // Adjust based on actual grid spacing

            std::vector<TReturn> dfdy(f.size(), 0);
            constexpr const label_t pad = order_ - 1;
            const label_t ny_padded = ny + 2 * pad;
            std::vector<double> padded_line(ny_padded, 0);

            for (label_t z = 0; z < nz; ++z)
            {
                for (label_t x = 0; x < nx; ++x)
                {
                    // Fill interior region of padded_line
                    for (label_t y = 0; y < ny; ++y)
                    {
                        padded_line[pad + y] = static_cast<double>(f[host::idxScalarGlobal(x, y, z, nx, ny)]);
                    }

                    // Set bottom ghost cells (reflect & negate)
                    for (label_t i = 0; i < pad; ++i)
                    {
                        padded_line[i] = -static_cast<double>(f[host::idxScalarGlobal(x, pad - i, z, nx, ny)]);
                    }

                    // Set top ghost cells (reflect & negate)
                    for (label_t i = 0; i < pad; ++i)
                    {
                        padded_line[pad + ny + i] = -static_cast<double>(f[host::idxScalarGlobal(x, ny - 2 - i, z, nx, ny)]);
                    }

                    // Compute derivatives for each point in y-direction
                    for (label_t y = 0; y < ny; ++y)
                    {
                        const label_t center = pad + y;

                        if constexpr (order_ == 2)
                        {
                            dfdy[host::idxScalarGlobal(x, y, z, nx, ny)] = static_cast<TReturn>(
                                (padded_line[center + 1] - padded_line[center - 1]) / (2.0 * static_cast<double>(dy)));
                        }

                        if constexpr (order_ == 4)
                        {
                            dfdy[host::idxScalarGlobal(x, y, z, nx, ny)] = static_cast<TReturn>(
                                (2.0 / 3.0 * (padded_line[center + 1] - padded_line[center - 1]) +
                                 1.0 / 12.0 * (padded_line[center + 2] - padded_line[center - 2])) /
                                static_cast<double>(dy));
                        }

                        if constexpr (order_ == 6)
                        {
                            dfdy[host::idxScalarGlobal(x, y, z, nx, ny)] = static_cast<TReturn>(
                                (3.0 / 4.0 * (padded_line[center + 1] - padded_line[center - 1]) +
                                 3.0 / 20.0 * (padded_line[center + 2] - padded_line[center - 2]) +
                                 1.0 / 60.0 * (padded_line[center + 3] - padded_line[center - 3])) /
                                static_cast<double>(dy));
                        }

                        if constexpr (order_ == 8)
                        {
                            dfdy[host::idxScalarGlobal(x, y, z, nx, ny)] = static_cast<TReturn>(
                                (4.0 / 5.0 * (padded_line[center + 1] - padded_line[center - 1]) +
                                 1.0 / 5.0 * (padded_line[center + 2] - padded_line[center - 2]) +
                                 4.0 / 105.0 * (padded_line[center + 3] - padded_line[center - 3]) +
                                 1.0 / 280.0 * (padded_line[center + 4] - padded_line[center - 4])) /
                                static_cast<double>(dy));
                        }
                    }
                }
            }
            return dfdy;

#endif
        }

        /**
         * @brief Calculates the z derivative of a scalar field
         * @return The z-derivative of f
         * @param f The field to be differentiated
         * @param mesh The lattice mesh
         **/
        template <const label_t order_, typename TReturn, typename T, class LatticeMesh>
        __host__ [[nodiscard]] const std::vector<TReturn> dfdz(
            const std::vector<T> &f,
            const LatticeMesh &mesh)
        {
            static_assert((order_ == 2) | (order_ == 4) | (order_ == 6) | (order_ == 8), "Invalid finite difference scheme order: valid orders are 2, 4, 6 and 8");

#ifdef MULTI_GPU

            static_assert(false, "derivative::dfdz not implemented for multi GPU yet");

#else

            const label_t nx = mesh.nx();
            const label_t ny = mesh.ny();
            const label_t nz = mesh.nz();
            constexpr const double dz = 1; // Adjust based on actual grid spacing

            std::vector<TReturn> dfdz(f.size(), 0);
            constexpr const label_t pad = order_ - 1;
            const label_t nz_padded = nz + 2 * pad;
            std::vector<double> padded_line(nz_padded, 0);

            for (label_t y = 0; y < ny; ++y)
            {
                for (label_t x = 0; x < nx; ++x)
                {
                    // Fill interior region of padded_line
                    for (label_t z = 0; z < nz; ++z)
                    {
                        padded_line[pad + z] = static_cast<double>(f[host::idxScalarGlobal(x, y, z, nx, ny)]);
                    }

                    // Set front ghost cells (reflect & negate)
                    for (label_t i = 0; i < pad; ++i)
                    {
                        padded_line[i] = -static_cast<double>(f[host::idxScalarGlobal(x, y, pad - i, nx, ny)]);
                    }

                    // Set back ghost cells (reflect & negate)
                    for (label_t i = 0; i < pad; ++i)
                    {
                        padded_line[pad + nz + i] = -static_cast<double>(f[host::idxScalarGlobal(x, y, nz - 2 - i, nx, ny)]);
                    }

                    // Compute derivatives for each point in z-direction
                    for (label_t z = 0; z < nz; ++z)
                    {
                        const label_t center = pad + z;

                        if constexpr (order_ == 2)
                        {
                            dfdz[host::idxScalarGlobal(x, y, z, nx, ny)] = static_cast<TReturn>(
                                (padded_line[center + 1] - padded_line[center - 1]) / (2.0 * static_cast<double>(dz)));
                        }

                        if constexpr (order_ == 4)
                        {
                            dfdz[host::idxScalarGlobal(x, y, z, nx, ny)] = static_cast<TReturn>(
                                (2.0 / 3.0 * (padded_line[center + 1] - padded_line[center - 1]) +
                                 1.0 / 12.0 * (padded_line[center + 2] - padded_line[center - 2])) /
                                static_cast<double>(dz));
                        }

                        if constexpr (order_ == 6)
                        {
                            dfdz[host::idxScalarGlobal(x, y, z, nx, ny)] = static_cast<TReturn>(
                                (3.0 / 4.0 * (padded_line[center + 1] - padded_line[center - 1]) +
                                 3.0 / 20.0 * (padded_line[center + 2] - padded_line[center - 2]) +
                                 1.0 / 60.0 * (padded_line[center + 3] - padded_line[center - 3])) /
                                static_cast<double>(dz));
                        }

                        if constexpr (order_ == 8)
                        {
                            dfdz[host::idxScalarGlobal(x, y, z, nx, ny)] = static_cast<TReturn>(
                                (4.0 / 5.0 * (padded_line[center + 1] - padded_line[center - 1]) +
                                 1.0 / 5.0 * (padded_line[center + 2] - padded_line[center - 2]) +
                                 4.0 / 105.0 * (padded_line[center + 3] - padded_line[center - 3]) +
                                 1.0 / 280.0 * (padded_line[center + 4] - padded_line[center - 4])) /
                                static_cast<double>(dz));
                        }
                    }
                }
            }

            return dfdz;

#endif
        }
    }
}

#include "curl.cuh"
#include "div.cuh"

#endif
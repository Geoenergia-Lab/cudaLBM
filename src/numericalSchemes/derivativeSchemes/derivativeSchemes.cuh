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
    namespace numericalSchemes
    {
        namespace derivative
        {
            __device__ __host__ [[nodiscard]] inline consteval host::label_t maxSchemeOrder() noexcept { return 8; }

            /**
             * @brief Calculates the x derivative of a scalar field
             * @return The x-derivative of f
             * @param[in] f The field to be differentiated
             * @param[in] mesh The lattice mesh
             **/
            template <const host::label_t SchemeOrder, typename TReturn, typename T>
            __host__ [[nodiscard]] const std::vector<TReturn> dfdx(
                const std::vector<T> &f,
                const host::latticeMesh &mesh)
            {
                LBM::numericalSchemes::assertions::validate<SchemeOrder, maxSchemeOrder()>();

                static_assert(MULTI_GPU_ASSERTION(), MULTI_GPU_MSG(derivative::dfdx));

                const host::label_t nx = mesh.dimension<axis::X>();
                const host::label_t ny = mesh.dimension<axis::Y>();
                const host::label_t nz = mesh.dimension<axis::Z>();

                const double dx = 1;

                std::vector<TReturn> dfdx(f.size(), 0);
                constexpr const host::label_t pad = SchemeOrder - 1;
                const host::label_t nx_padded = nx + 2 * pad;
                std::vector<double> padded_line(nx_padded, 0);

                for (host::label_t z = 0; z < nz; ++z)
                {
                    for (host::label_t y = 0; y < ny; ++y)
                    {
                        // Fill interior region of padded_line
                        for (host::label_t x = 0; x < nx; ++x)
                        {
                            padded_line[pad + x] = static_cast<double>(f[global::idx(x, y, z, nx, ny)]);
                        }

                        // Set left ghost cells (reflect & negate)
                        for (host::label_t i = 0; i < pad; ++i)
                        {
                            padded_line[i] = static_cast<double>(-f[global::idx(pad - i, y, z, nx, ny)]);
                        }

                        // Set right ghost cells (reflect & negate)
                        for (host::label_t i = 0; i < pad; ++i)
                        {
                            padded_line[pad + nx + i] = static_cast<double>(-f[global::idx(nx - 2 - i, y, z, nx, ny)]);
                        }

                        // Compute derivatives for each point in x-direction
                        for (host::label_t x = 0; x < nx; ++x)
                        {
                            const host::label_t center = pad + x;

                            if constexpr (SchemeOrder == 2)
                            {
                                dfdx[global::idx(x, y, z, nx, ny)] = static_cast<TReturn>(
                                    (padded_line[center + 1] - padded_line[center - 1]) / (2.0 * static_cast<double>(dx)));
                            }

                            if constexpr (SchemeOrder == 4)
                            {
                                dfdx[global::idx(x, y, z, nx, ny)] = static_cast<TReturn>(
                                    (2.0 / 3.0 * (padded_line[center + 1] - padded_line[center - 1]) +
                                     1.0 / 12.0 * (padded_line[center + 2] - padded_line[center - 2])) /
                                    static_cast<double>(dx));
                            }

                            if constexpr (SchemeOrder == 6)
                            {
                                dfdx[global::idx(x, y, z, nx, ny)] = static_cast<TReturn>(
                                    (3.0 / 4.0 * (padded_line[center + 1] - padded_line[center - 1]) +
                                     3.0 / 20.0 * (padded_line[center + 2] - padded_line[center - 2]) +
                                     1.0 / 60.0 * (padded_line[center + 3] - padded_line[center - 3])) /
                                    static_cast<double>(dx));
                            }

                            if constexpr (SchemeOrder == 8)
                            {
                                dfdx[global::idx(x, y, z, nx, ny)] = static_cast<TReturn>(
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
            }

            /**
             * @brief Calculates the y derivative of a scalar field
             * @return The y-derivative of f
             * @param[in] f The field to be differentiated
             * @param[in] mesh The lattice mesh
             **/
            template <const host::label_t SchemeOrder, typename TReturn, typename T>
            __host__ [[nodiscard]] const std::vector<TReturn> dfdy(
                const std::vector<T> &f,
                const host::latticeMesh &mesh)
            {
                LBM::numericalSchemes::assertions::validate<SchemeOrder, maxSchemeOrder()>();

                static_assert(MULTI_GPU_ASSERTION(), MULTI_GPU_MSG(derivative::dfdy));

                const host::label_t nx = mesh.dimension<axis::X>();
                const host::label_t ny = mesh.dimension<axis::Y>();
                const host::label_t nz = mesh.dimension<axis::Z>();
                constexpr const double dy = 1; // Adjust based on actual grid spacing

                std::vector<TReturn> dfdy(f.size(), 0);
                constexpr const host::label_t pad = SchemeOrder - 1;
                const host::label_t ny_padded = ny + 2 * pad;
                std::vector<double> padded_line(ny_padded, 0);

                for (host::label_t z = 0; z < nz; ++z)
                {
                    for (host::label_t x = 0; x < nx; ++x)
                    {
                        // Fill interior region of padded_line
                        for (host::label_t y = 0; y < ny; ++y)
                        {
                            padded_line[pad + y] = static_cast<double>(f[global::idx(x, y, z, nx, ny)]);
                        }

                        // Set bottom ghost cells (reflect & negate)
                        for (host::label_t i = 0; i < pad; ++i)
                        {
                            padded_line[i] = -static_cast<double>(f[global::idx(x, pad - i, z, nx, ny)]);
                        }

                        // Set top ghost cells (reflect & negate)
                        for (host::label_t i = 0; i < pad; ++i)
                        {
                            padded_line[pad + ny + i] = -static_cast<double>(f[global::idx(x, ny - 2 - i, z, nx, ny)]);
                        }

                        // Compute derivatives for each point in y-direction
                        for (host::label_t y = 0; y < ny; ++y)
                        {
                            const host::label_t center = pad + y;

                            if constexpr (SchemeOrder == 2)
                            {
                                dfdy[global::idx(x, y, z, nx, ny)] = static_cast<TReturn>(
                                    (padded_line[center + 1] - padded_line[center - 1]) / (2.0 * static_cast<double>(dy)));
                            }

                            if constexpr (SchemeOrder == 4)
                            {
                                dfdy[global::idx(x, y, z, nx, ny)] = static_cast<TReturn>(
                                    (2.0 / 3.0 * (padded_line[center + 1] - padded_line[center - 1]) +
                                     1.0 / 12.0 * (padded_line[center + 2] - padded_line[center - 2])) /
                                    static_cast<double>(dy));
                            }

                            if constexpr (SchemeOrder == 6)
                            {
                                dfdy[global::idx(x, y, z, nx, ny)] = static_cast<TReturn>(
                                    (3.0 / 4.0 * (padded_line[center + 1] - padded_line[center - 1]) +
                                     3.0 / 20.0 * (padded_line[center + 2] - padded_line[center - 2]) +
                                     1.0 / 60.0 * (padded_line[center + 3] - padded_line[center - 3])) /
                                    static_cast<double>(dy));
                            }

                            if constexpr (SchemeOrder == 8)
                            {
                                dfdy[global::idx(x, y, z, nx, ny)] = static_cast<TReturn>(
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
            }

            /**
             * @brief Calculates the z derivative of a scalar field
             * @return The z-derivative of f
             * @param[in] f The field to be differentiated
             * @param[in] mesh The lattice mesh
             **/
            template <const host::label_t SchemeOrder, typename TReturn, typename T>
            __host__ [[nodiscard]] const std::vector<TReturn> dfdz(
                const std::vector<T> &f,
                const host::latticeMesh &mesh)
            {
                LBM::numericalSchemes::assertions::validate<SchemeOrder, maxSchemeOrder()>();

                static_assert(MULTI_GPU_ASSERTION(), MULTI_GPU_MSG(derivative::dfdz));

                const host::label_t nx = mesh.dimension<axis::X>();
                const host::label_t ny = mesh.dimension<axis::Y>();
                const host::label_t nz = mesh.dimension<axis::Z>();
                constexpr const double dz = 1; // Adjust based on actual grid spacing

                std::vector<TReturn> dfdz(f.size(), 0);
                constexpr const host::label_t pad = SchemeOrder - 1;
                const host::label_t nz_padded = nz + 2 * pad;
                std::vector<double> padded_line(nz_padded, 0);

                for (host::label_t y = 0; y < ny; ++y)
                {
                    for (host::label_t x = 0; x < nx; ++x)
                    {
                        // Fill interior region of padded_line
                        for (host::label_t z = 0; z < nz; ++z)
                        {
                            padded_line[pad + z] = static_cast<double>(f[global::idx(x, y, z, nx, ny)]);
                        }

                        // Set front ghost cells (reflect & negate)
                        for (host::label_t i = 0; i < pad; ++i)
                        {
                            padded_line[i] = -static_cast<double>(f[global::idx(x, y, pad - i, nx, ny)]);
                        }

                        // Set back ghost cells (reflect & negate)
                        for (host::label_t i = 0; i < pad; ++i)
                        {
                            padded_line[pad + nz + i] = -static_cast<double>(f[global::idx(x, y, nz - 2 - i, nx, ny)]);
                        }

                        // Compute derivatives for each point in z-direction
                        for (host::label_t z = 0; z < nz; ++z)
                        {
                            const host::label_t center = pad + z;

                            if constexpr (SchemeOrder == 2)
                            {
                                dfdz[global::idx(x, y, z, nx, ny)] = static_cast<TReturn>(
                                    (padded_line[center + 1] - padded_line[center - 1]) / (2.0 * static_cast<double>(dz)));
                            }

                            if constexpr (SchemeOrder == 4)
                            {
                                dfdz[global::idx(x, y, z, nx, ny)] = static_cast<TReturn>(
                                    (2.0 / 3.0 * (padded_line[center + 1] - padded_line[center - 1]) +
                                     1.0 / 12.0 * (padded_line[center + 2] - padded_line[center - 2])) /
                                    static_cast<double>(dz));
                            }

                            if constexpr (SchemeOrder == 6)
                            {
                                dfdz[global::idx(x, y, z, nx, ny)] = static_cast<TReturn>(
                                    (3.0 / 4.0 * (padded_line[center + 1] - padded_line[center - 1]) +
                                     3.0 / 20.0 * (padded_line[center + 2] - padded_line[center - 2]) +
                                     1.0 / 60.0 * (padded_line[center + 3] - padded_line[center - 3])) /
                                    static_cast<double>(dz));
                            }

                            if constexpr (SchemeOrder == 8)
                            {
                                dfdz[global::idx(x, y, z, nx, ny)] = static_cast<TReturn>(
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
            }
        }
    }
}

#include "curl.cuh"
#include "div.cuh"

#endif
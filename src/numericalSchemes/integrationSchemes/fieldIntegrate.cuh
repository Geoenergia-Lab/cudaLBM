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
    Field integration schemes

Namespace
    LBM

SourceFiles
    fieldIntegrate.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_FIELDINTEGRATE_CUH
#define __MBLBM_FIELDINTEGRATE_CUH

namespace LBM
{
    /**
     * @brief Calculates the integral of a scalar field along the x-axis.
     * @tparam order_ The order of the integration scheme. Currently, only 2nd order is implemented.
     * @return The integrated field of f with respect to x.
     * @param f The field to be integrated.
     * @param mesh The lattice mesh.
     * @note This function uses the cumulative trapezoidal rule. The integration constant is set by
     * assuming the integral is zero at x=0 for each (y, z) line.
     **/
    template <const label_t order_, typename TReturn, typename T, class LatticeMesh>
    __host__ [[nodiscard]] const std::vector<TReturn> integrate_x(
        const std::vector<T> &f,
        const LatticeMesh &mesh)
    {
        static_assert(order_ == 2, "Invalid integration scheme order: only 2nd order (Trapezoidal Rule) is currently implemented.");

#ifdef MULTI_GPU

        static_assert(false, "fieldIntegrate not implemented for multi GPU yet");

#else

        const label_t nx = mesh.nx();
        const label_t ny = mesh.ny();
        const label_t nz = mesh.nz();
        constexpr const double dx = 1.0;

        std::vector<TReturn> integral_f(f.size(), 0);

        for (label_t z = 0; z < nz; ++z)
        {
            for (label_t y = 0; y < ny; ++y)
            {
                // Initial condition for integration along this x-line
                integral_f[host::idxScalarGlobal(0, y, z, nx, ny)] = 0;

                // Cumulative integration using the trapezoidal rule
                for (label_t x = 1; x < nx; ++x)
                {
                    const label_t current_idx = host::idxScalarGlobal(x, y, z, nx, ny);
                    const label_t prev_idx = host::idxScalarGlobal(x - 1, y, z, nx, ny);

                    integral_f[current_idx] = integral_f[prev_idx] + static_cast<TReturn>(0.5 * dx * (static_cast<double>(f[prev_idx]) + static_cast<double>(f[current_idx])));
                }
            }
        }
        return integral_f;

#endif
    }

    /**
     * @brief Calculates the integral of a scalar field along the y-axis.
     **/
    template <const label_t order_, typename TReturn, typename T, class LatticeMesh>
    __host__ [[nodiscard]] const std::vector<TReturn> integrate_y(
        const std::vector<T> &f,
        const LatticeMesh &mesh)
    {
        static_assert(order_ == 2, "Invalid integration scheme order: only 2nd order (Trapezoidal Rule) is currently implemented.");

#ifdef MULTI_GPU

        static_assert(false, "fieldIntegrate not implemented for multi GPU yet");

#else

        const label_t nx = mesh.nx();
        const label_t ny = mesh.ny();
        const label_t nz = mesh.nz();
        constexpr const double dy = 1.0;

        std::vector<TReturn> integral_f(f.size(), 0);

        for (label_t z = 0; z < nz; ++z)
        {
            for (label_t x = 0; x < nx; ++x)
            {
                // Initial condition for integration along this y-line
                integral_f[host::idxScalarGlobal(x, 0, z, nx, ny)] = 0;

                // Cumulative integration using the trapezoidal rule
                for (label_t y = 1; y < ny; ++y)
                {
                    const label_t current_idx = host::idxScalarGlobal(x, y, z, nx, ny);
                    const label_t prev_idx = host::idxScalarGlobal(x, y - 1, z, nx, ny);

                    integral_f[current_idx] = integral_f[prev_idx] + static_cast<TReturn>(0.5 * dy * (static_cast<double>(f[prev_idx]) + static_cast<double>(f[current_idx])));
                }
            }
        }
        return integral_f;

#endif
    }

    /**
     * @brief Calculates the integral of a scalar field along the z-axis.
     **/
    template <const label_t order_, typename TReturn, typename T, class LatticeMesh>
    __host__ [[nodiscard]] const std::vector<TReturn> integrate_z(
        const std::vector<T> &f,
        const LatticeMesh &mesh)
    {
        static_assert(order_ == 2, "Invalid integration scheme order: only 2nd order (Trapezoidal Rule) is currently implemented.");

#ifdef MULTI_GPU

        static_assert(false, "fieldIntegrate not implemented for multi GPU yet");

#else

        const label_t nx = mesh.nx();
        const label_t ny = mesh.ny();
        const label_t nz = mesh.nz();
        constexpr const double dz = 1.0;

        std::vector<TReturn> integral_f(f.size(), 0);

        for (label_t y = 0; y < ny; ++y)
        {
            for (label_t x = 0; x < nx; ++x)
            {
                // Initial condition for integration along this z-line
                integral_f[host::idxScalarGlobal(x, y, 0, nx, ny)] = 0;

                // Cumulative integration using the trapezoidal rule
                for (label_t z = 1; z < nz; ++z)
                {
                    const label_t current_idx = host::idxScalarGlobal(x, y, z, nx, ny);
                    const label_t prev_idx = host::idxScalarGlobal(x, y, z - 1, nx, ny);

                    integral_f[current_idx] = integral_f[prev_idx] + static_cast<TReturn>(0.5 * dz * (static_cast<double>(f[prev_idx]) + static_cast<double>(f[current_idx])));
                }
            }
        }
        return integral_f;

#endif
    }
}

#endif // __MBLBM_FIELDINTEGRATE_CUH
/*---------------------------------------------------------------------------*\
|                                                                             |
| HermiteLBM: CUDA-based moment representation Lattice Boltzmann Method       |
| Developed at UDESC - State University of Santa Catarina                     |
| Website: https://www.udesc.br                                               |
| Github: https://github.com/Geoenergia-Lab/HermiteLBM                        |
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
    This file is part of HermiteLBM.

    HermiteLBM is free software: you can redistribute it and/or modify it
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
    Configuration of the main GPU kernel

Namespace
    LBM::host, LBM::device

SourceFiles
    launchConfig.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_MOMENTBASEDLBM_LAUNCHCONFIG_CUH
#define __MBLBM_MOMENTBASEDLBM_LAUNCHCONFIG_CUH

namespace LBM
{
    /**
     * @brief Minimum number of blocks per streaming microprocessor
     * @tparam VelocitySet The velocity set in use
     **/
    template <class VelocitySet>
    __device__ __host__ [[nodiscard]] inline consteval device::label_t MIN_BLOCKS_PER_MP() noexcept
    {
        // D3Q19 thermal model
        if constexpr (std::is_same_v<VelocitySet, D3Q19<Thermal>>)
        {
            return 1;
        }

        // D3Q19 isothermal model
        if constexpr (std::is_same_v<VelocitySet, D3Q19<Isothermal>>)
        {
            return 1;
        }

        // D3Q27 thermal model
        if constexpr (std::is_same_v<VelocitySet, D3Q27<Thermal>>)
        {
            return 1;
        }

        // D3Q27 isothermal model
        if constexpr (std::is_same_v<VelocitySet, D3Q27<Isothermal>>)
        {
            return 1;
        }

        // Default
        return 1;
    }

    /**
     * @brief Use experimental block co-operative halo saving
     **/
    __device__ __host__ [[nodiscard]] inline consteval bool use_cooperative_halo() noexcept
    {
#ifdef USE_SMEM_HALO
#if USE_SMEM_HALO == true
        return true;
#elif USE_SMEM_HALO == false
        return false;
#endif
#else
        return false;
#endif
    }

    /**
     * @brief Runtime bounds checking for GPU kernels
     **/
    __device__ __host__ [[nodiscard]] inline consteval bool out_of_bounds_check() noexcept
    {
#ifdef OOB_CHECK
        return true;
#else
        return false;
#endif
    }
}

#endif
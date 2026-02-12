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
    A class handling the device halo. This class is used to exchange the
    microscopic velocity components at the edge of a CUDA block

Namespace
    LBM::device

SourceFiles
    haloSharedMemoryOperations.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_HALOSHAREDMEMORYOPERATIONS_CUH
#define __MBLBM_HALOSHAREDMEMORYOPERATIONS_CUH

#ifdef MULTI_GPU_HALO_SHARED_MEMORY_OPERATIONS

/**
 * @brief Transposes the block halo into the shared memory
 * @param[in] pop Array containing the populations for the particular thread
 * @param[out] s_buffer Shared array containing the packed population halos
 *
 * This device function saves population values to halo regions for
 * neighboring blocks to read.
 **/
template <const label_t N>
__device__ static inline void transpose_to_shared(
    const thread::array<scalar_t, VelocitySet::Q()> &pop,
    thread::array<scalar_t, N> &s_buffer) noexcept
{
    const label_t x = threadIdx.x + block::nx() * blockIdx.x;
    const label_t y = threadIdx.y + block::ny() * blockIdx.y;
    const label_t z = threadIdx.z + block::nz() * blockIdx.z;

    // Calculate base indices for each boundary type
    constexpr label_t x_size = block::ny() * block::nz();
    constexpr label_t y_size = block::nx() * block::nz();
    constexpr label_t z_size = block::nx() * block::ny();

    // West boundary (5 populations)
    if (West(x))
    {
        const label_t base_idx = threadIdx.y + threadIdx.z * block::ny();
        s_buffer[base_idx + (0 * x_size) + 0] = pop[q_i<2>()];
        s_buffer[base_idx + (1 * x_size) + 0] = pop[q_i<8>()];
        s_buffer[base_idx + (2 * x_size) + 0] = pop[q_i<10>()];
        s_buffer[base_idx + (3 * x_size) + 0] = pop[q_i<14>()];
        s_buffer[base_idx + (4 * x_size) + 0] = pop[q_i<16>()];
    }

    // East boundary (5 populations)
    if (East(x))
    {
        const label_t base_idx = threadIdx.y + threadIdx.z * block::ny();
        constexpr label_t east_offset = 5 * x_size;
        s_buffer[east_offset + base_idx + (0 * x_size) + 0] = pop[q_i<1>()];
        s_buffer[east_offset + base_idx + (1 * x_size) + 0] = pop[q_i<7>()];
        s_buffer[east_offset + base_idx + (2 * x_size) + 0] = pop[q_i<9>()];
        s_buffer[east_offset + base_idx + (3 * x_size) + 1] = pop[q_i<13>()];
        s_buffer[east_offset + base_idx + (4 * x_size) + 1] = pop[q_i<15>()];
    }

    // South boundary (5 populations)
    if (South(y))
    {
        const label_t base_idx = threadIdx.x + threadIdx.z * block::nx();
        constexpr label_t south_offset = 10 * x_size;
        s_buffer[south_offset + base_idx + (0 * y_size) + 1] = pop[q_i<4>()];
        s_buffer[south_offset + base_idx + (1 * y_size) + 1] = pop[q_i<8>()];
        s_buffer[south_offset + base_idx + (2 * y_size) + 1] = pop[q_i<12>()];
        s_buffer[south_offset + base_idx + (3 * y_size) + 1] = pop[q_i<13>()];
        s_buffer[south_offset + base_idx + (4 * y_size) + 1] = pop[q_i<18>()];
    }

    // North boundary (5 populations)
    if (North(y))
    {
        const label_t base_idx = threadIdx.x + threadIdx.z * block::nx();
        constexpr label_t north_offset = 10 * x_size + 5 * y_size;
        s_buffer[north_offset + base_idx + (0 * y_size) + 1] = pop[q_i<3>()];
        s_buffer[north_offset + base_idx + (1 * y_size) + 2] = pop[q_i<7>()];
        s_buffer[north_offset + base_idx + (2 * y_size) + 2] = pop[q_i<11>()];
        s_buffer[north_offset + base_idx + (3 * y_size) + 2] = pop[q_i<14>()];
        s_buffer[north_offset + base_idx + (4 * y_size) + 2] = pop[q_i<17>()];
    }

    // Back boundary (5 populations)
    if (Back(z))
    {
        const label_t base_idx = threadIdx.x + threadIdx.y * block::nx();
        constexpr label_t back_offset = 10 * x_size + 10 * y_size;
        s_buffer[back_offset + base_idx + (0 * z_size) + 2] = pop[q_i<6>()];
        s_buffer[back_offset + base_idx + (1 * z_size) + 2] = pop[q_i<10>()];
        s_buffer[back_offset + base_idx + (2 * z_size) + 2] = pop[q_i<12>()];
        s_buffer[back_offset + base_idx + (3 * z_size) + 2] = pop[q_i<15>()];
        s_buffer[back_offset + base_idx + (4 * z_size) + 3] = pop[q_i<17>()];
    }

    // Front boundary (5 populations)
    if (Front(z))
    {
        const label_t base_idx = threadIdx.x + threadIdx.y * block::nx();
        constexpr label_t front_offset = 10 * x_size + 10 * y_size + 5 * z_size;
        s_buffer[front_offset + base_idx + (0 * z_size) + 3] = pop[q_i<5>()];
        s_buffer[front_offset + base_idx + (1 * z_size) + 3] = pop[q_i<9>()];
        s_buffer[front_offset + base_idx + (2 * z_size) + 3] = pop[q_i<11>()];
        s_buffer[front_offset + base_idx + (3 * z_size) + 3] = pop[q_i<16>()];
        s_buffer[front_offset + base_idx + (4 * z_size) + 3] = pop[q_i<18>()];
    }

    __syncthreads();
}

/**
 * @brief Saves population data to halo regions for neighboring blocks
 * @param[in] s_buffer Shared array containing the packed population halos
 * @param[out] gGhost Collection of pointers to the halo faces
 *
 * This device function saves population values to halo regions for
 * neighboring blocks to read.
 **/
template <const label_t N>
__device__ static inline void save_from_shared(
    const thread::array<scalar_t, N> &s_buffer,
    const device::ptrCollection<6, scalar_t> &gGhost) noexcept
{
    const label_t warpId = warpID(threadIdx.x, threadIdx.y, threadIdx.z);
    const label_t offset = block::warp_size() * (warpId % 2);
    const label_t idx_in_warp = idxWarp(threadIdx.x, threadIdx.y, threadIdx.z);

    // Equivalent of threadIdx.alpha, threadIdx.beta
    const dim2 xy = ij<axis::X, axis::Y>(idx_in_warp + offset);
    const dim2 xz = ij<axis::X, axis::Z>(idx_in_warp + offset);
    const dim2 yz = ij<axis::Y, axis::Z>(idx_in_warp + offset);

    const label_t ID = idx_block(threadIdx.x, threadIdx.y, threadIdx.z);

    constexpr label_t padded_stride = block::size() + 1; // 513 instead of 512
    const scalar_t val0 = s_buffer[ID];
    const scalar_t val1 = s_buffer[ID + padded_stride];
    const scalar_t val2 = s_buffer[ID + (2 * padded_stride)];
    const scalar_t val3 = s_buffer[ID + (3 * padded_stride)];

    switch (warpId / 2)
    {
    case 0:
    {
        gGhost.ptr<0>()[idxPopX<0, VelocitySet::QF()>(yz.i, yz.j, blockIdx)] = val0;
        gGhost.ptr<1>()[idxPopX<3, VelocitySet::QF()>(yz.i, yz.j, blockIdx)] = val1;
        gGhost.ptr<3>()[idxPopY<1, VelocitySet::QF()>(xz.i, xz.j, blockIdx)] = val2;
        gGhost.ptr<4>()[idxPopZ<4, VelocitySet::QF()>(xy.i, xy.j, blockIdx)] = val3;

        break;
    }
    case 1:
    {
        gGhost.ptr<0>()[idxPopX<1, VelocitySet::QF()>(yz.i, yz.j, blockIdx)] = val0;
        gGhost.ptr<1>()[idxPopX<4, VelocitySet::QF()>(yz.i, yz.j, blockIdx)] = val1;
        gGhost.ptr<3>()[idxPopY<2, VelocitySet::QF()>(xz.i, xz.j, blockIdx)] = val2;
        gGhost.ptr<5>()[idxPopZ<0, VelocitySet::QF()>(xy.i, xy.j, blockIdx)] = val3;

        break;
    }
    case 2:
    {
        gGhost.ptr<0>()[idxPopX<2, VelocitySet::QF()>(yz.i, yz.j, blockIdx)] = val0;
        gGhost.ptr<2>()[idxPopY<0, VelocitySet::QF()>(xz.i, xz.j, blockIdx)] = val1;
        gGhost.ptr<3>()[idxPopY<3, VelocitySet::QF()>(xz.i, xz.j, blockIdx)] = val2;
        gGhost.ptr<5>()[idxPopZ<1, VelocitySet::QF()>(xy.i, xy.j, blockIdx)] = val3;

        break;
    }
    case 3:
    {
        gGhost.ptr<0>()[idxPopX<3, VelocitySet::QF()>(yz.i, yz.j, blockIdx)] = val0;
        gGhost.ptr<2>()[idxPopY<1, VelocitySet::QF()>(xz.i, xz.j, blockIdx)] = val1;
        gGhost.ptr<3>()[idxPopY<4, VelocitySet::QF()>(xz.i, xz.j, blockIdx)] = val2;
        gGhost.ptr<5>()[idxPopZ<2, VelocitySet::QF()>(xy.i, xy.j, blockIdx)] = val3;

        break;
    }
    case 4:
    {
        gGhost.ptr<0>()[idxPopX<4, VelocitySet::QF()>(yz.i, yz.j, blockIdx)] = val0;
        gGhost.ptr<2>()[idxPopY<2, VelocitySet::QF()>(xz.i, xz.j, blockIdx)] = val1;
        gGhost.ptr<4>()[idxPopZ<0, VelocitySet::QF()>(xy.i, xy.j, blockIdx)] = val2;
        gGhost.ptr<5>()[idxPopZ<3, VelocitySet::QF()>(xy.i, xy.j, blockIdx)] = val3;

        break;
    }
    case 5:
    {
        gGhost.ptr<1>()[idxPopX<0, VelocitySet::QF()>(yz.i, yz.j, blockIdx)] = val0;
        gGhost.ptr<2>()[idxPopY<3, VelocitySet::QF()>(xz.i, xz.j, blockIdx)] = val1;
        gGhost.ptr<4>()[idxPopZ<1, VelocitySet::QF()>(xy.i, xy.j, blockIdx)] = val2;
        gGhost.ptr<5>()[idxPopZ<4, VelocitySet::QF()>(xy.i, xy.j, blockIdx)] = val3;

        break;
    }
    case 6:
    {
        gGhost.ptr<1>()[idxPopX<1, VelocitySet::QF()>(yz.i, yz.j, blockIdx)] = val0;
        gGhost.ptr<2>()[idxPopY<4, VelocitySet::QF()>(xz.i, xz.j, blockIdx)] = val1;
        gGhost.ptr<4>()[idxPopZ<2, VelocitySet::QF()>(xy.i, xy.j, blockIdx)] = val2;

        break;
    }
    case 7:
    {
        gGhost.ptr<1>()[idxPopX<2, VelocitySet::QF()>(yz.i, yz.j, blockIdx)] = val0;
        gGhost.ptr<3>()[idxPopY<0, VelocitySet::QF()>(xz.i, xz.j, blockIdx)] = val1;
        gGhost.ptr<4>()[idxPopZ<3, VelocitySet::QF()>(xy.i, xy.j, blockIdx)] = val2;

        break;
    }
    }
}

#endif

#endif
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
    Global utility functions and structures for LBM simulations.
    Contains core functionalities used throughout the LBM implementation.

Namespace
    LBM

SourceFiles
    globalFunctions.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_GLOBALFUNCTIONS_CUH
#define __MBLBM_GLOBALFUNCTIONS_CUH

#include "LBMIncludes.cuh"
#include "LBMTypedefs.cuh"

namespace LBM
{
    /**
     * @brief Compile-time recursive loop unroller
     * @tparam Start Starting index (inclusive)
     * @tparam End Ending index (exclusive)
     * @tparam F Callable type accepting integral_constant<label_t>
     * @param f Function object to execute per iteration
     *
     * @note Equivalent to runtime loop: `for(label_t i=Start; i<End; ++i)`
     * @note Enables `if constexpr` usage in loop bodies
     * @warning Recursion depth limited by compiler constraints
     *
     * Example usage:
     * @code
     * constexpr_for<0, 5>([](auto i) {
     *     // i is integral_constant<label_t, N>
     *     if constexpr (i.value % 2 == 0) { ... }
     * });
     * @endcode
     **/
    namespace host
    {
        template <const label_t Start, const label_t End, typename F>
        __host__ inline constexpr void constexpr_for(F &&f)
        {
            if constexpr (Start < End)
            {
                f(std::integral_constant<label_t, Start>());
                if constexpr (Start + 1 < End)
                {
                    host::constexpr_for<Start + 1, End>(std::forward<F>(f));
                }
            }
        }
    }

    namespace device
    {
        template <const label_t Start, const label_t End, typename F>
        __device__ inline constexpr void constexpr_for(F &&f)
        {
            if constexpr (Start < End)
            {
                f(integralConstant<label_t, Start>());
                if constexpr (Start + 1 < End)
                {
                    device::constexpr_for<Start + 1, End>(std::forward<F>(f));
                }
            }
        }
    }

    /**
     * @brief Number of hydrodynamic moments
     **/
    __device__ __host__ [[nodiscard]] inline consteval label_t NUMBER_MOMENTS() { return 10; }

    /**
     * @brief Host-side indexing operations
     **/
    namespace host
    {
        /**
         * @brief Compute moment memory index
         * @tparam mom Moment index [0, NUMBER_MOMENTS())
         * @param tx,ty,tz Thread-local coordinates
         * @param bx,by,bz Block indices
         * @param nxBlocks Number of blocks in x-direction
         * @param nyBlocks Number of blocks in y-direction
         * @return Linearized index in moment array
         *
         * Layout: [bx][by][bz][tz][ty][tx][mom] (mom fastest varying)
         **/
        template <const label_t mom>
        __host__ [[nodiscard]] inline label_t idxMom(
            const label_t tx, const label_t ty, const label_t tz,
            const label_t bx, const label_t by, const label_t bz,
            const label_t nxBlocks, const label_t nyBlocks) noexcept
        {
            return mom + NUMBER_MOMENTS() * (tx + block::nx() * (ty + block::ny() * (tz + block::nz() * (bx + nxBlocks * (by + nyBlocks * bz)))));
        }

        /**
         * @overload Compute moment memory index
         * @tparam mom Moment index [0, NUMBER_MOMENTS())
         * @param tx,ty,tz Thread-local coordinates
         * @param bx,by,bz Block indices
         * @param nxBlocks Number of blocks in x-direction
         * @param nyBlocks Number of blocks in y-direction
         * @return Linearized index in moment array
         *
         * Layout: [bx][by][bz][tz][ty][tx][mom] (mom fastest varying)
         **/
        template <const label_t mom, class M>
        __host__ [[nodiscard]] inline label_t idxMom(
            const label_t tx, const label_t ty, const label_t tz,
            const label_t bx, const label_t by, const label_t bz,
            const M &mesh) noexcept
        {
            return idxMom<mom>(tx, ty, tz, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks());
        }

        /**
         * @overload Run-time indexing for the moment variable
         * @param mom Moment index [0, NUMBER_MOMENTS())
         * @param tx,ty,tz Thread-local coordinates
         * @param bx,by,bz Block indices
         * @param nxBlocks Number of blocks in x-direction
         * @param nyBlocks Number of blocks in y-direction
         * @return Linearized index in moment array
         *
         * Layout: [bx][by][bz][tz][ty][tx][mom] (mom fastest varying)
         **/
        __host__ [[nodiscard]] inline label_t idxMom(
            const label_t tx, const label_t ty, const label_t tz,
            const label_t bx, const label_t by, const label_t bz,
            const label_t mom, const label_t nxBlocks, const label_t nyBlocks) noexcept
        {
            return mom + NUMBER_MOMENTS() * (tx + block::nx() * (ty + block::ny() * (tz + block::nz() * (bx + nxBlocks * (by + nyBlocks * bz)))));
        }

        /**
         * @overload Run-time indexing for the moment variable
         * @param mom Moment index [0, NUMBER_MOMENTS())
         * @param tx,ty,tz Thread-local coordinates
         * @param bx,by,bz Block indices
         * @param nxBlocks Number of blocks in x-direction
         * @param nyBlocks Number of blocks in y-direction
         * @return Linearized index in moment array
         *
         * Layout: [bx][by][bz][tz][ty][tx][mom] (mom fastest varying)
         **/
        template <class M>
        __host__ [[nodiscard]] inline label_t idxMom(
            const label_t tx, const label_t ty, const label_t tz,
            const label_t bx, const label_t by, const label_t bz,
            const label_t mom, const M &mesh) noexcept
        {
            return idxMom(tx, ty, tz, bx, by, bz, mom, mesh.nxBlocks(), mesh.nyBlocks());
        }

        /**
         * @brief Memory index (host version)
         * @param tx,ty,tz Thread-local coordinates
         * @param bx,by,bz Block indices
         * @param nxBlocks,nyBlocks Number of blocks in the x and y directions
         * @return Linearized index using mesh constants
         *
         * Layout: [bx][by][bz][tz][ty][tx] (tx fastest varying)
         **/
        __host__ [[nodiscard]] inline label_t idx(
            const label_t tx, const label_t ty, const label_t tz,
            const label_t bx, const label_t by, const label_t bz,
            const label_t nxBlocks, const label_t nyBlocks) noexcept
        {
            return (tx + block::nx() * (ty + block::ny() * (tz + block::nz() * (bx + nxBlocks * (by + nyBlocks * bz)))));
        }

        /**
         * @brief Memory index (host version)
         * @param tx,ty,tz Thread-local coordinates
         * @param bx,by,bz Block indices
         * @param mesh The mesh
         * @return Linearized index using mesh constants
         *
         * Layout: [bx][by][bz][tz][ty][tx] (tx fastest varying)
         **/
        template <class M>
        __host__ [[nodiscard]] inline label_t idx(
            const label_t tx, const label_t ty, const label_t tz,
            const label_t bx, const label_t by, const label_t bz,
            const M &mesh) noexcept
        {
            return idx(tx, ty, tz, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks());
        }

        /**
         * @brief Global scalar field index (collapsed 3D)
         * @param x,y,z Global coordinates
         * @param nx,ny Global dimensions
         * @return Linearized index: x + nx*(y + ny*z)
         **/
        __host__ [[nodiscard]] inline label_t idxScalarGlobal(
            const label_t x, const label_t y, const label_t z,
            const label_t nx, const label_t ny) noexcept
        {
            return x + (nx * (y + (ny * z)));
        }

        /**
         * @brief Index for X-aligned population arrays
         * @tparam pop Population index
         * @tparam QF Number of populations
         * @param ty,tz Thread-local y/z coordinates
         * @param bx,by,bz Block indices
         * @param nxBlocks Number of blocks in x-direction
         * @param nyBlocks Number of blocks in y-direction
         * @return Linearized index: ty + block::ny()*(tz + block::nz()*(pop + QF*(bx + nxBlocks*(by + nyBlocks*bz)))
         * @note Optimized for coalesced memory access in X-direction
         **/
        template <const label_t pop, const label_t QF>
        __host__ [[nodiscard]] inline label_t idxPopX(
            const label_t ty, const label_t tz,
            const label_t bx, const label_t by, const label_t bz,
            const label_t nxBlocks, const label_t nyBlocks) noexcept
        {
            return ty + block::ny() * (tz + block::nz() * (pop + QF * (bx + nxBlocks * (by + nyBlocks * bz))));
        }

        /**
         * @brief Index for Y-aligned population arrays
         * @copydetails idxPopX
         * @param tx,tz Thread-local x/z coordinates
         * @return Linearized index: tx + block::nx()*(tz + block::nz()*(pop + QF*(bx + nxBlocks*(by + nyBlocks*bz)))
         **/
        template <const label_t pop, const label_t QF>
        __host__ [[nodiscard]] inline label_t idxPopY(
            const label_t tx, const label_t tz,
            const label_t bx, const label_t by, const label_t bz,
            const label_t nxBlocks, const label_t nyBlocks) noexcept
        {
            return tx + block::nx() * (tz + block::nz() * (pop + QF * (bx + nxBlocks * (by + nyBlocks * bz))));
        }

        /**
         * @brief Index for Z-aligned population arrays
         * @copydetails idxPopX
         * @param tx,ty Thread-local x/y coordinates
         * @return Linearized index: tx + block::nx()*(ty + block::ny()*(pop + QF*(bx + nxBlocks*(by + nyBlocks*bz)))
         **/
        template <const label_t pop, const label_t QF>
        __host__ [[nodiscard]] inline label_t idxPopZ(
            const label_t tx, const label_t ty,
            const label_t bx, const label_t by, const label_t bz,
            const label_t nxBlocks, const label_t nyBlocks) noexcept
        {
            return tx + block::nx() * (ty + block::ny() * (pop + QF * (bx + nxBlocks * (by + nyBlocks * bz))));
        }

        template <const axisDirection alpha, const label_t pop, const label_t QF>
        __host__ [[nodiscard]] inline label_t idxPop(
            const label_t tx, const label_t ty, const label_t tz,
            const label_t bx, const label_t by, const label_t bz,
            const label_t nxBlocks, const label_t nyBlocks)
        {
            if constexpr (alpha == X)
            {
                return idxPopX<pop, QF>(ty, tz, bx, by, bz, nxBlocks, nyBlocks);
            }

            if constexpr (alpha == Y)
            {
                return idxPopY<pop, QF>(tx, tz, bx, by, bz, nxBlocks, nyBlocks);
            }

            if constexpr (alpha == Z)
            {
                return idxPopZ<pop, QF>(tx, ty, bx, by, bz, nxBlocks, nyBlocks);
            }
        }
    }

    /**
     * @brief Device-side indexing operations
     **/
    namespace device
    {
        /**
         * @brief Check if current thread exceeds global bounds
         * @note Uses device constants device::nx, device::ny, device::nz
         * @return True if thread is outside domain boundaries
         **/
        __device__ [[nodiscard]] inline bool out_of_bounds() noexcept
        {
            return ((threadIdx.x + blockDim.x * blockIdx.x >= device::nx) || (threadIdx.y + blockDim.y * blockIdx.y >= device::ny) || (threadIdx.z + blockDim.z * blockIdx.z >= device::nz));
        }

        /**
         * @brief Moment memory index (device version)
         * @tparam mom Moment index [0, NUMBER_MOMENTS())
         * @param tx,ty,tz Thread-local coordinates
         * @param bx,by,bz Block indices
         * @return Linearized index using device constants device::NUM_BLOCK_X/Y
         *
         * Layout: [bx][by][bz][tz][ty][tx][mom] (mom fastest varying)
         **/
        template <const label_t mom>
        __device__ [[nodiscard]] inline label_t idxMom(const label_t tx, const label_t ty, const label_t tz, const label_t bx, const label_t by, const label_t bz) noexcept
        {
            return mom + NUMBER_MOMENTS() * (tx + block::nx() * (ty + block::ny() * (tz + block::nz() * (bx + device::NUM_BLOCK_X * (by + device::NUM_BLOCK_Y * bz)))));
        }

        /**
         * @overload
         * @param tx Thread coordinates (dim3)
         * @param bx Block indices (dim3)
         **/
        template <const label_t mom>
        __device__ [[nodiscard]] inline label_t idxMom(const dim3 &tx, const dim3 &bx) noexcept
        {
            return idxMom<mom>(tx.x, tx.y, tx.z, bx.x, bx.y, bx.z);
        }

        /**
         * @brief Memory index (device version)
         * @param tx,ty,tz Thread-local coordinates
         * @param bx,by,bz Block indices
         * @return Linearized index using device constants device::NUM_BLOCK_X/Y
         *
         * Layout: [bx][by][bz][tz][ty][tx] (tx fastest varying)
         **/
        __device__ [[nodiscard]] inline label_t idx(
            const label_t tx, const label_t ty, const label_t tz,
            const label_t bx, const label_t by, const label_t bz) noexcept
        {
            return (tx + block::nx() * (ty + block::ny() * (tz + block::nz() * (bx + NUM_BLOCK_X * (by + NUM_BLOCK_Y * bz)))));
        }

        /**
         * @overload
         * @param tx Thread coordinates (dim3)
         * @param bx Block indices (dim3)
         **/
        __device__ [[nodiscard]] inline label_t idx(const dim3 tx, const dim3 bx) noexcept
        {
            return idx(tx.x, tx.y, tx.z, bx.x, bx.y, bx.z);
        }

        /**
         * @overload
         **/
        __device__ [[nodiscard]] inline label_t idx() noexcept
        {
            return idx(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);
        }

        /**
         * @brief Memory index within a block (device version)
         * @param tx Thread-local x-coordinate
         * @param ty Thread-local y-coordinate
         * @param tz Thread-local z-coordinate
         * @return Linearized index using block dimensions (block::nx() and block::ny())
         *
         * Layout within a block: [tz][ty][tx] (tx fastest varying)
         * Strides:
         *   - x-stride: 1
         *   - y-stride: block::nx()
         *   - z-stride: block::nx() * block::ny()
         **/
        __device__ [[nodiscard]] inline label_t idxBlock(const label_t tx, const label_t ty, const label_t tz) noexcept
        {
            return tx + block::nx() * (ty + block::ny() * tz);
        }

        /**
         * @overload
         * @param tx Thread coordinates (dim3)
         **/
        __device__ [[nodiscard]] inline label_t idxBlock(const dim3 &tx) noexcept
        {
            return idxBlock(tx.x, tx.y, tx.z);
        }

        /**
         * @overload
         **/
        __device__ [[nodiscard]] inline label_t idxBlock() noexcept
        {
            return idxBlock(threadIdx.x, threadIdx.y, threadIdx.z);
        }

        /**
         * @brief Population index for X-aligned arrays (device version)
         * @tparam pop Population index
         * @tparam QF Number of populations
         * @param ty,tz Thread-local y/z coordinates
         * @param bx,by,bz Block indices
         * @return Linearized index: ty + block::ny()*(tz + block::nz()*(pop + QF*(bx + device::NUM_BLOCK_X*(by + device::NUM_BLOCK_Y*bz)))
         **/
        template <const label_t pop, const label_t QF>
        __device__ [[nodiscard]] inline label_t idxPopX(const label_t ty, const label_t tz, const label_t bx, const label_t by, const label_t bz) noexcept
        {
            return ty + block::ny() * (tz + block::nz() * (pop + QF * (bx + device::NUM_BLOCK_X * (by + device::NUM_BLOCK_Y * bz))));
        }

        /**
         * @overload
         * @param ty,tz Thread-local y/z coordinates
         * @param bx Block indices (dim3)
         **/
        template <const label_t pop, const label_t QF>
        __device__ [[nodiscard]] inline label_t idxPopX(const label_t ty, const label_t tz, const dim3 &bx) noexcept
        {
            return idxPopX<pop, QF>(ty, tz, bx.x, bx.y, bx.z);
        }

        /**
         * @brief Population index for Y-aligned arrays (device version)
         * @copydetails idxPopX
         * @param tx,tz Thread-local x/z coordinates
         * @return Linearized index: tx + block::nx()*(tz + block::nz()*(pop + QF*(bx + device::NUM_BLOCK_X*(by + device::NUM_BLOCK_Y*bz)))
         **/
        template <const label_t pop, const label_t QF>
        __device__ [[nodiscard]] inline label_t idxPopY(const label_t tx, const label_t tz, const label_t bx, const label_t by, const label_t bz) noexcept
        {
            return tx + block::nx() * (tz + block::nz() * (pop + QF * (bx + device::NUM_BLOCK_X * (by + device::NUM_BLOCK_Y * bz))));
        }

        /**
         * @overload
         * @param tx,tz Thread-local x/z coordinates
         * @param bx Block indices (dim3)
         **/
        template <const label_t pop, const label_t QF>
        __device__ [[nodiscard]] inline label_t idxPopY(const label_t tx, const label_t tz, const dim3 &bx) noexcept
        {
            return idxPopY<pop, QF>(tx, tz, bx.x, bx.y, bx.z);
        }

        /**
         * @brief Population index for Z-aligned arrays (device version)
         * @copydetails idxPopX
         * @param tx,ty Thread-local x/y coordinates
         * @return Linearized index: tx + block::nx()*(ty + block::ny()*(pop + QF*(bx + device::NUM_BLOCK_X*(by + device::NUM_BLOCK_Y*bz)))
         **/
        template <const label_t pop, const label_t QF>
        __device__ [[nodiscard]] inline label_t idxPopZ(const label_t tx, const label_t ty, const label_t bx, const label_t by, const label_t bz) noexcept
        {
            return tx + block::nx() * (ty + block::ny() * (pop + QF * (bx + device::NUM_BLOCK_X * (by + device::NUM_BLOCK_Y * bz))));
        }

        /**
         * @overload
         * @param tx,ty Thread-local x/y coordinates
         * @param bx Block indices (dim3)
         **/
        template <const label_t pop, const label_t QF>
        __device__ [[nodiscard]] inline label_t idxPopZ(const label_t tx, const label_t ty, const dim3 &bx) noexcept
        {
            return idxPopZ<pop, QF>(tx, ty, bx.x, bx.y, bx.z);
        }
    }

    /**
     * @brief Moment indices for hydrodynamic variables
     **/
    namespace index
    {
        __device__ __host__ [[nodiscard]] inline consteval label_t rho() { return 0; } // < Density
        __device__ __host__ [[nodiscard]] inline consteval label_t u() { return 1; }   // < X-velocity
        __device__ __host__ [[nodiscard]] inline consteval label_t v() { return 2; }   // < Y-velocity
        __device__ __host__ [[nodiscard]] inline consteval label_t w() { return 3; }   // < Z-velocity
        __device__ __host__ [[nodiscard]] inline consteval label_t xx() { return 4; }  // < XX-stress component
        __device__ __host__ [[nodiscard]] inline consteval label_t xy() { return 5; }  // < XY-stress component
        __device__ __host__ [[nodiscard]] inline consteval label_t xz() { return 6; }  // < XZ-stress component
        __device__ __host__ [[nodiscard]] inline consteval label_t yy() { return 7; }  // < YY-stress component
        __device__ __host__ [[nodiscard]] inline consteval label_t yz() { return 8; }  // < YZ-stress component
        __device__ __host__ [[nodiscard]] inline consteval label_t zz() { return 9; }  // < ZZ-stress component
    }

    /**
     * @brief Reference density 1.0
     **/
    template <typename T>
    __device__ __host__ [[nodiscard]] inline consteval T rho0() noexcept
    {
        return 1.0;
    }

    /**
     * @brief Queries a device and gets its properties
     * @param[in] deviceID The ID of the device to query
     * @return A cudaDeviceProp struct containing the properties of deviceID
     **/
    __host__ [[nodiscard]] const cudaDeviceProp getDeviceProperties(const int deviceID)
    {
        cudaDeviceProp props;

        if (cudaGetDeviceProperties(&props, deviceID) != cudaSuccess)
        {
            throw std::runtime_error("Failed to get CUDA device properties");
        }

        return props;
    }

    /**
     * @brief Allocates a symbol of type T to the device
     * @param[in] symbol The symbol to which the value is to be copied
     * @param[in] value The value to copy to the symbol
     **/
    template <typename T>
    void copyToSymbol(const T &symbol, const T value, const label_t index)
    {
        cudaDeviceSynchronize();
        const T valueTemp = value;
        checkCudaErrors(cudaMemcpyToSymbol(symbol, &valueTemp, sizeof(T), static_cast<std::size_t>(index) * sizeof(T), cudaMemcpyHostToDevice));
        cudaDeviceSynchronize();
    }

    template <typename T>
    void copyToSymbol(const T &symbol, const T value)
    {
        cudaDeviceSynchronize();
        const T valueTemp = value;
        checkCudaErrors(cudaMemcpyToSymbol(symbol, &valueTemp, sizeof(T), 0, cudaMemcpyHostToDevice));
        cudaDeviceSynchronize();
    }

    template <typename T, const std::size_t N>
    void copyToSymbol(const T (&symbol)[N], const T (&value)[N])
    {
        cudaDeviceSynchronize();
        checkCudaErrors(cudaMemcpyToSymbol(symbol, value, N * sizeof(T), 0, cudaMemcpyHostToDevice));
        cudaDeviceSynchronize();
    }

    template <typename T, const std::size_t N>
    void copyToSymbol(const T (&symbol)[N], const T value, const label_t index)
    {
        if (index >= N)
        {
            throw std::runtime_error("Error setting device symbol index" + std::to_string(index) + " out of bounds for array of size " + std::to_string(N) + ".");
        }
        cudaDeviceSynchronize();
        const T valueTemp = value;
        checkCudaErrors(cudaMemcpyToSymbol(symbol, &valueTemp, sizeof(T), static_cast<std::size_t>(index) * sizeof(T), cudaMemcpyHostToDevice));
        cudaDeviceSynchronize();
    }

    /**
     * @brief Allocates a symbol of type T to the device
     * @param[in] func A pointer to the kernel to configure
     * @param[in] functionName The name of the function
     **/
    template <const label_t smem_alloc_size, class T>
    __host__ void kernelSetup(T *func, const std::string &functionName) noexcept
    {
        checkCudaErrors(cudaFuncSetCacheConfig(func, cudaFuncCachePreferShared));
        checkCudaErrors(cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_alloc_size));
        std::cout << "Allocating " << smem_alloc_size << " bytes of dynamic shared memory to " << functionName << " kernel" << std::endl;
        std::cout << std::endl;
    }

    /**
     * @brief Allocates a symbol of type T to the device
     * @param[in] func A pointer to the kernel to configure
     **/
    template <const label_t smem_alloc_size, class T>
    __host__ void kernelSetup(T *func) noexcept
    {
        checkCudaErrors(cudaFuncSetCacheConfig(func, cudaFuncCachePreferShared));
        checkCudaErrors(cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_alloc_size));
    }
}

#endif
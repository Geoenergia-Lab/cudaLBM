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
    A class holding information about the solution grid

Namespace
    LBM::host

SourceFiles
    latticeMesh.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_LATTICEMESH_CUH
#define __MBLBM_LATTICEMESH_CUH

#include "../LBMIncludes.cuh"
#include "../LBMTypedefs.cuh"
#include "../globalFunctions.cuh"
#include "../programControl/programControl.cuh"

namespace LBM
{
    namespace host
    {
        /**
         * @class latticeMesh
         * @brief Represents the computational grid for LBM simulations
         *
         * This class encapsulates the 3D lattice grid information including
         * dimensions, block decomposition, and physical properties. It handles
         * initialization from configuration files, validation of grid parameters,
         * and synchronization of grid properties with GPU device memory.
         **/
        class latticeMesh
        {
        public:
            /**
             * @brief Constructs a lattice mesh from program configuration
             * @param[in] programCtrl Program control object containing simulation parameters
             * @throws Error if mesh dimensions are invalid or GPU memory is insufficient
             *
             * This constructor reads mesh dimensions from the "programControl" file and performs:
             * - Validation of block decomposition compatibility
             * - Memory requirement checking for GPU
             * - Calculation of LBM relaxation parameters
             * - Initialization of device constants for GPU execution
             **/
            __host__ [[nodiscard]] latticeMesh(const programControl &programCtrl) noexcept
                : nx_(string::extractParameter<label_t>(string::readFile("latticeMesh"), "nx")),
                  ny_(string::extractParameter<label_t>(string::readFile("latticeMesh"), "ny")),
                  nz_(string::extractParameter<label_t>(string::readFile("latticeMesh"), "nz")),
                  nPoints_(nx_ * ny_ * nz_),
                  L_(
                      {string::extractParameter<scalar_t>(string::readFile("latticeMesh"), "Lx"),
                       string::extractParameter<scalar_t>(string::readFile("latticeMesh"), "Ly"),
                       string::extractParameter<scalar_t>(string::readFile("latticeMesh"), "Lz")})
            {
                std::cout << "latticeMesh:" << std::endl;
                std::cout << "{" << std::endl;
                std::cout << "    nx = " << nx_ << ";" << std::endl;
                std::cout << "    ny = " << ny_ << ";" << std::endl;
                std::cout << "    nz = " << nz_ << ";" << std::endl;
                std::cout << "    Lx = " << L_.x << ";" << std::endl;
                std::cout << "    Ly = " << L_.y << ";" << std::endl;
                std::cout << "    Lz = " << L_.z << ";" << std::endl;
                std::cout << "};" << std::endl;
                std::cout << std::endl;

                std::cout << "block:" << std::endl;
                std::cout << "{" << std::endl;
                std::cout << "    nx = " << block::nx() << ";" << std::endl;
                std::cout << "    ny = " << block::ny() << ";" << std::endl;
                std::cout << "    nz = " << block::nz() << ";" << std::endl;
                std::cout << "};" << std::endl;
                std::cout << std::endl;

                // Perform a block dimensions safety check
                {
                    if (!(block::nx() * nxBlocks() == nx_))
                    {
                        errorHandler(ERR_SIZE, "block::nx() * mesh.nxBlocks() not equal to mesh.nx()\nMesh dimensions should be multiples of 8");
                    }
                    if (!(block::ny() * nyBlocks() == ny_))
                    {
                        errorHandler(ERR_SIZE, "block::ny() * mesh.nyBlocks() not equal to mesh.ny()\nMesh dimensions should be multiples of 8");
                    }
                    if (!(block::nz() * nzBlocks() == nz_))
                    {
                        errorHandler(ERR_SIZE, "block::nz() * mesh.nzBlocks() not equal to mesh.nz()\nMesh dimensions should be multiples of 8");
                    }
                    if (!(block::nx() * nxBlocks() * block::ny() * nyBlocks() * block::nz() * nzBlocks() == nx_ * ny_ * nz_))
                    {
                        errorHandler(ERR_SIZE, "block::nx() * nxBlocks() * block::ny() * nyBlocks() * block::nz() * nzBlocks() not equal to mesh.nPoints()\nMesh dimensions should be multiples of 8");
                    }
                }

                // Safety check for the mesh dimensions
                {
                    const uint64_t nxTemp = static_cast<uint64_t>(nx_);
                    const uint64_t nyTemp = static_cast<uint64_t>(ny_);
                    const uint64_t nzTemp = static_cast<uint64_t>(nz_);
                    const uint64_t nPointsTemp = nxTemp * nyTemp * nzTemp;
                    constexpr const uint64_t typeLimit = static_cast<uint64_t>(std::numeric_limits<label_t>::max());

                    // Check that the mesh dimensions won't overflow the type limit for label_t
                    if (nPointsTemp >= typeLimit)
                    {
                        errorHandler(ERR_SIZE,
                                     "\nMesh size exceeds maximum allowed value:\n"
                                     "Number of mesh points: " +
                                         std::to_string(nPointsTemp) +
                                         "\nLimit of label_t: " +
                                         std::to_string(typeLimit));
                    }

                    // Check that the mesh dimensions are not too large for GPU memory
                    {
                        const cudaDeviceProp props = getDeviceProperties(programCtrl.deviceList()[0]);
                        const uint64_t totalMemTemp = static_cast<uint64_t>(props.totalGlobalMem);
                        const uint64_t allocationSize = nPointsTemp * static_cast<uint64_t>(sizeof(scalar_t)) * static_cast<uint64_t>(NUMBER_MOMENTS());

                        if (allocationSize >= totalMemTemp)
                        {
                            const double gbAllocation = static_cast<double>(allocationSize / (1024 * 1024 * 1024));
                            const double gbAvailable = static_cast<double>(totalMemTemp / (1024 * 1024 * 1024));

                            errorHandler(ERR_SIZE,
                                         "\nInsufficient GPU memory:\n"
                                         "Attempted to allocate: " +
                                             std::to_string(allocationSize) +
                                             " bytes (" + std::to_string(gbAllocation) + " GB)\n"
                                                                                         "Available GPU memory: " +
                                             std::to_string(totalMemTemp) +
                                             " bytes (" + std::to_string(gbAvailable) + " GB)");
                        }
                    }
                }

                // Allocate programControl symbols on the GPU (clean up later)
                {
                    const scalar_t viscosityTemp = programCtrl.u_inf() * programCtrl.L_char() / programCtrl.Re();
                    const scalar_t tauTemp = static_cast<scalar_t>(0.5) + static_cast<scalar_t>(3.0) * viscosityTemp;
                    const scalar_t omegaTemp = static_cast<scalar_t>(1.0) / tauTemp;
                    const scalar_t t_omegaVarTemp = static_cast<scalar_t>(1) - omegaTemp;
                    const scalar_t omegaVar_d2Temp = omegaTemp * static_cast<scalar_t>(0.5);

                    copyToSymbol(device::L_char, programCtrl.L_char());
                    copyToSymbol(device::Re, programCtrl.Re());
                    copyToSymbol(device::tau, tauTemp);
                    copyToSymbol(device::omega, omegaTemp);
                    copyToSymbol(device::t_omegaVar, t_omegaVarTemp);
                    copyToSymbol(device::omegaVar_d2, omegaVar_d2Temp);
                }

                // Allocate mesh symbols on the GPU
                copyToSymbol(device::nx, nx_);
                copyToSymbol(device::ny, ny_);
                copyToSymbol(device::nz, nz_);
                copyToSymbol(device::NUM_BLOCK_X, nxBlocks());
                copyToSymbol(device::NUM_BLOCK_Y, nyBlocks());
                copyToSymbol(device::NUM_BLOCK_Z, nzBlocks());
            };

            // Constructor to initialise a cut plane
            __host__ [[nodiscard]] latticeMesh(const label_t nx, const label_t ny, const label_t nz) noexcept
                : nx_(nx),
                  ny_(ny),
                  nz_(nz),
                  nPoints_(nx_ * ny_ * nz_),
                  L_(
                      {string::extractParameter<scalar_t>(string::readFile("latticeMesh"), "Lx"),
                       string::extractParameter<scalar_t>(string::readFile("latticeMesh"), "Ly"),
                       string::extractParameter<scalar_t>(string::readFile("latticeMesh"), "Lz")}){};

            __host__ [[nodiscard]] latticeMesh(const blockLabel_t meshDimensions) noexcept
                : nx_(meshDimensions.nx),
                  ny_(meshDimensions.ny),
                  nz_(meshDimensions.nz),
                  nPoints_(nx_ * ny_ * nz_),
                  L_(
                      {string::extractParameter<scalar_t>(string::readFile("latticeMesh"), "Lx"),
                       string::extractParameter<scalar_t>(string::readFile("latticeMesh"), "Ly"),
                       string::extractParameter<scalar_t>(string::readFile("latticeMesh"), "Lz")}){};

            __host__ [[nodiscard]] latticeMesh(const host::latticeMesh &mesh, const blockLabel_t meshDimensions) noexcept
                : nx_(meshDimensions.nx),
                  ny_(meshDimensions.ny),
                  nz_(meshDimensions.nz),
                  nPoints_(nx_ * ny_ * nz_),
                  L_(mesh.L()){};

            __host__ [[nodiscard]] latticeMesh(const host::latticeMesh &mesh) noexcept
                : nx_(mesh.nx()),
                  ny_(mesh.ny()),
                  nz_(mesh.nz()),
                  nPoints_(nx_ * ny_ * nz_),
                  L_(mesh.L()){};

            /**
             * @name Grid Dimension Accessors
             * @brief Provide access to grid dimensions
             * @return Dimension value in specified direction
             **/
            __device__ __host__ [[nodiscard]] inline constexpr label_t nx() const noexcept
            {
                return nx_;
            }
            __device__ __host__ [[nodiscard]] inline constexpr label_t ny() const noexcept
            {
                return ny_;
            }
            __device__ __host__ [[nodiscard]] inline constexpr label_t nz() const noexcept
            {
                return nz_;
            }
            __device__ __host__ [[nodiscard]] inline constexpr label_t nPoints() const noexcept
            {
                return nPoints_;
            }

            /**
             * @name Block Decomposition Accessors
             * @brief Provide access to CUDA block decomposition
             * @return Number of blocks in specified direction
             **/
            __device__ __host__ [[nodiscard]] inline constexpr label_t nxBlocks() const noexcept
            {
                return nx_ / block::nx();
            }
            __device__ __host__ [[nodiscard]] inline constexpr label_t nyBlocks() const noexcept
            {
                return ny_ / block::ny();
            }
            __device__ __host__ [[nodiscard]] inline constexpr label_t nzBlocks() const noexcept
            {
                return nz_ / block::nz();
            }
            __device__ __host__ [[nodiscard]] inline constexpr label_t nBlocks() const noexcept
            {
                return (nx_ / block::nx()) * (ny_ / block::ny()) * (nz_ / block::nz());
            }

            /**
             * @brief Get thread block dimensions for CUDA kernel launches
             * @return dim3 structure with thread block dimensions
             **/
            __device__ __host__ [[nodiscard]] static inline consteval dim3 threadBlock() noexcept
            {
                return {block::nx(), block::ny(), block::nz()};
            }

            /**
             * @brief Get grid dimensions for CUDA kernel launches
             * @return dim3 structure with grid dimensions
             **/
            __device__ __host__ [[nodiscard]] inline constexpr dim3 gridBlock() const noexcept
            {
                return {static_cast<uint32_t>(nx_ / block::nx()), static_cast<uint32_t>(ny_ / block::ny()), static_cast<uint32_t>(nz_ / block::nz())};
            }

            /**
             * @brief Get physical domain dimensions
             * @return Const reference to pointVector containing domain size
             **/
            __host__ [[nodiscard]] inline constexpr const pointVector &L() const noexcept
            {
                return L_;
            }

            /**
             * @brief Get the number of physical dimensions of the mesh
             * @return Const reference to pointVector containing domain size
             **/
            __host__ [[nodiscard]] inline constexpr label_t nDims() const noexcept
            {
                return static_cast<label_t>(nx_ > 1) + static_cast<label_t>(ny_ > 1) + static_cast<label_t>(nz_ > 1);
            }

            /**
             * @brief Boundary check for the faces
             * @param x,y,z The coordinate of the point
             * @return True if the point is on the boundary, false otherwise
             **/
            __host__ [[nodiscard]] inline constexpr bool West(const label_t x) const noexcept
            {
                return (x == 0);
            }
            __host__ [[nodiscard]] inline constexpr bool East(const label_t x) const noexcept
            {
                return (x == nx_ - 1);
            }
            __host__ [[nodiscard]] inline constexpr bool South(const label_t y) const noexcept
            {
                return (y == 0);
            }
            __host__ [[nodiscard]] inline constexpr bool North(const label_t y) const noexcept
            {
                return (y == ny_ - 1);
            }
            __host__ [[nodiscard]] inline constexpr bool Back(const label_t z) const noexcept
            {
                return (z == 0);
            }
            __host__ [[nodiscard]] inline constexpr bool Front(const label_t z) const noexcept
            {
                return (z == nz_ - 1);
            }

        private:
            /**
             * @brief The number of lattices in the x, y and z directions
             **/
            const label_t nx_;
            const label_t ny_;
            const label_t nz_;
            const label_t nPoints_;

            /**
             * @brief Physical dimensions of the domain
             **/
            const pointVector L_;
        };
    }
}

#include "deviceLatticeMesh.cuh"

#endif
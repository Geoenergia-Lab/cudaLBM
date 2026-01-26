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
    A class holding information about the solution grid on the GPU(s)

Namespace
    LBM::device

SourceFiles
    deviceLatticeMesh.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_DEVICELATTICEMESH_CUH
#define __MBLBM_DEVICELATTICEMESH_CUH

namespace LBM
{
    namespace device
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
            __host__ [[nodiscard]] inline constexpr latticeMesh(
                const label_t deviceID,
                const label_t nxPoints, const label_t nyPoints, const label_t nzPoints) noexcept
                : deviceID_(deviceID),
                  nx_(nxPoints),
                  ny_(nyPoints),
                  nz_(nzPoints),
                  nPoints_(nx_ * ny_ * nz_),
                  nxBlocks_(nxPoints / block::nx()),
                  nyBlocks_(nyPoints / block::ny()),
                  nzBlocks_(nzPoints / block::nz()),
                  bx0_(0),
                  by0_(0),
                  bz0_(0){};

            __host__ [[nodiscard]] inline latticeMesh(
                const label_t deviceID,
                const label_t nxPoints, const label_t nyPoints, const label_t nzPoints,
                const label_t bx0, const label_t by0, const label_t bz0) noexcept
                : deviceID_(deviceID),
                  nx_(nxPoints),
                  ny_(nyPoints),
                  nz_(nzPoints),
                  nPoints_(nx_ * ny_ * nz_),
                  nxBlocks_(nxPoints / block::nx()),
                  nyBlocks_(nyPoints / block::ny()),
                  nzBlocks_(nzPoints / block::nz()),
                  bx0_(bx0),
                  by0_(by0),
                  bz0_(bz0)
            {

                if (!(block::nx() * nxBlocks_ == nx_))
                {
                    errorHandler(ERR_SIZE, "block::nx() * mesh.nxBlocks() not equal to mesh.nx()\nMesh dimensions should be multiples of 8");
                }
                if (!(block::ny() * nyBlocks_ == ny_))
                {
                    errorHandler(ERR_SIZE, "block::ny() * mesh.nyBlocks() not equal to mesh.ny()\nMesh dimensions should be multiples of 8");
                }
                if (!(block::nz() * nzBlocks_ == nz_))
                {
                    errorHandler(ERR_SIZE, "block::nz() * mesh.nzBlocks() not equal to mesh.nz()\nMesh dimensions should be multiples of 8");
                }
                if (!(block::nx() * nxBlocks_ * block::ny() * nyBlocks_ * block::nz() * nzBlocks_ == nx_ * ny_ * nz_))
                {
                    errorHandler(ERR_SIZE, "block::nx() * nxBlocks() * block::ny() * nyBlocks() * block::nz() * nzBlocks() not equal to mesh.nPoints()\nMesh dimensions should be multiples of 8");
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
                        // const cudaDeviceProp props = getDeviceProperties(programCtrl.deviceList()[0]);
                        std::cout << "CLEAN UP THIS PART: MAKE SURE YOU SET THE RIGHT DEVICE ACCORDING TO DEVICEID" << std::endl;
                        const cudaDeviceProp props = getDeviceProperties(0);

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
            };

            inline void print() const noexcept
            {
                std::cout << "device[" << deviceID_ << "]::latticeMesh:" << std::endl;
                std::cout << "{" << std::endl;
                std::cout << "\tnx = " << nx_ << " [" << (bx0_ * block::nx()) << " <= x <= " << ((bx0_ * block::nx()) + nx_) - 1 << "]" << std::endl;
                std::cout << "\tny = " << ny_ << " [" << (by0_ * block::ny()) << " <= y <= " << ((by0_ * block::ny()) + ny_) - 1 << "]" << std::endl;
                std::cout << "\tnz = " << nz_ << " [" << (bz0_ * block::nz()) << " <= z <= " << ((bz0_ * block::nz()) + nz_) - 1 << "]" << std::endl;
                std::cout << "};" << std::endl;
            }

        private:
            const label_t deviceID_;

            /**
             * @brief The number of lattices in the x, y and z directions
             **/
            const label_t nx_;
            const label_t ny_;
            const label_t nz_;
            const label_t nPoints_;

            const label_t nxBlocks_;
            const label_t nyBlocks_;
            const label_t nzBlocks_;

            /**
             * @brief Global block offsets: the blocks at which this partition of the mesh begins
             **/
            const label_t bx0_;
            const label_t by0_;
            const label_t bz0_;
        };
    }
}

#endif
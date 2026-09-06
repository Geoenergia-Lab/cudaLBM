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
    A templated class for allocating collections of arrays on the CPU

Namespace
    LBM::host

SourceFiles
    hostArrayCollection.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_HOSTARRAYCOLLECTION_CUH
#define __MBLBM_HOSTARRAYCOLLECTION_CUH

namespace LBM
{
    namespace host
    {
        /**
         * @class arrayCollection
         * @brief Templated container for multiple field arrays with flexible initialization
         * @tparam T Data type of array elements
         * @tparam cType Constructor type specification
         **/
        template <typename T>
        class arrayCollection
        {
        public:
            __host__ [[nodiscard]] arrayCollection(
                const name_t &fileName,
                const words_t &varNames)
                : empty_(!(std::filesystem::exists(fileName))),
                  arr_(initialiseVector(fileName, empty_)),
                  varNames_(varNames) {}

            /**
             * @brief Destructor for the host arrayCollection class
             **/
            __host__ ~arrayCollection() {}

            /**
             * @brief Get read-only access to underlying data
             * @return Const reference to data vector
             **/
            __host__ [[nodiscard]] inline constexpr const std::vector<T> &arr() const noexcept
            {
                return arr_;
            }

            /**
             * @brief Get variable names in collection
             * @return Const reference to variable names vector
             **/
            __host__ [[nodiscard]] inline const words_t &varNames() const noexcept
            {
                return varNames_;
            }

            /**
             * @brief Check if the collection is empty (i.e., if the file was not found)
             * @return True if empty, false otherwise
             **/
            __host__ [[nodiscard]] inline constexpr bool empty() const noexcept
            {
                return empty_;
            }

            /**
             * @brief Splits the underlying large vector into N fields
             * @param[in] mesh The lattice mesh
             * @return Vector of vectors where each inner vector contains all values for one variable
             * @throws std::invalid_argument if input size doesn't match mesh dimensions
             *
             * This function reorganizes data from AoS format (where all variables for
             * each point are stored together) to SoA format (where each variable's values
             * are stored in separate contiguous arrays).
             **/
            template <const bool Deinterleave, const bool Sort>
            __host__ [[nodiscard]] const std::vector<std::vector<T>> splitFields(const host::latticeMesh &mesh) const
            {
                if constexpr (Deinterleave)
                {
                    static_assert(Sort == false, "Cannot de-interleave and sort fields at the same time; this makes no sense!");
                }

                const host::label_t nNodes = mesh.dimension<axis::X>() * mesh.dimension<axis::Y>() * mesh.dimension<axis::Z>();

                if (arr().size() % nNodes != 0)
                {
                    throw std::invalid_argument("fMom size (" + std::to_string(arr().size()) + ") is not divisible by mesh points (" + std::to_string(nNodes) + ")");
                }

                const host::label_t nFields = arr().size() / nNodes;
                std::vector<std::vector<T>> soa(nFields, std::vector<T>(nNodes, 0));

                // We are not deinterleaving, so we can just do a straightforward copy
                if constexpr (!Deinterleave)
                {
                    for (host::label_t field = 0; field < nFields; field++)
                    {
                        std::memcpy(soa[field].data(), std::addressof(arr()[field * nNodes]), nNodes * sizeof(T));
                    }

                    // We are sorting, so use std::sort
                    if constexpr (Sort)
                    {
                        for (std::vector<T> &fieldVec : soa)
                        {
                            std::sort(fieldVec.begin(), fieldVec.end());
                        }
                    }

                    return soa;
                }

                const host::label_t nxGPUs = mesh.nDevices<axis::X>();
                const host::label_t nyGPUs = mesh.nDevices<axis::Y>();
                const host::label_t nzGPUs = mesh.nDevices<axis::Z>();

                const host::label_t nxBlocksPerDevice = mesh.nBlocks<axis::X>() / nxGPUs;
                const host::label_t nyBlocksPerDevice = mesh.nBlocks<axis::Y>() / nyGPUs;
                const host::label_t nzBlocksPerDevice = mesh.nBlocks<axis::Z>() / nzGPUs;

                constexpr const host::label_t pointsPerBlock = block::size<host::label_t>();
                const host::label_t nPointsPerDevice = nxBlocksPerDevice * nyBlocksPerDevice * nzBlocksPerDevice * pointsPerBlock;

                GPU::forAll(
                    mesh.nDevices(),
                    [&](const host::label_t GPU_x, const host::label_t GPU_y, const host::label_t GPU_z)
                    {
                        const host::label_t virtualDeviceIndex = GPU::idx(GPU_x, GPU_y, GPU_z, nxGPUs, nyGPUs);

                        host::forAll(
                            mesh.blocksPerDevice(),
                            [&](const host::label_t bx, const host::label_t by, const host::label_t bz,
                                const host::label_t tx, const host::label_t ty, const host::label_t tz)
                            {
                                // Local (GPU‑order) index
                                const host::label_t localIdx = host::idx(tx, ty, tz, bx, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>());

                                // Fill all fields
                                for (host::label_t field = 0; field < nFields; ++field)
                                {
                                    const host::label_t srcIdx = field * nNodes + virtualDeviceIndex * nPointsPerDevice + localIdx;
                                    soa[field][destIndex<Deinterleave>(mesh, localIdx, tx, ty, tz, bx, by, bz, GPU_x, GPU_y, GPU_z, nxBlocksPerDevice, nyBlocksPerDevice, nzBlocksPerDevice)] = arr()[srcIdx];
                                }
                            });
                    });

                return soa;
            }

            /**
             * @brief Split the fields into separate vectors and de-interleave into natural Cartesian coordinates
             **/
            __host__ [[nodiscard]] const std::vector<std::vector<T>> deinterleaveAoS(const host::latticeMesh &mesh) const
            {
                return splitFields<true, false>(mesh);
            }

            /**
             * @brief Split the fields into separate vectors without de-interleaving
             **/
            __host__ [[nodiscard]] const std::vector<std::vector<T>> splitFieldsRaw(const host::latticeMesh &mesh) const
            {
                return splitFields<false, false>(mesh);
            }

            /**
             * @brief Split the fields into separate vectors and sort each field in ascending order
             **/
            __host__ [[nodiscard]] const std::vector<std::vector<T>> splitFieldsAndSort(const host::latticeMesh &mesh) const
            {
                return splitFields<false, true>(mesh);
            }

        private:
            /**
             * @brief Flag indicating whether the collection is empty (i.e., if the file was not found)
             **/
            const bool empty_;

            /**
             * @brief The underlying std::vector
             **/
            const std::vector<T> arr_;

            /**
             * @brief Names of the solution variables
             **/
            const words_t varNames_;

            /**
             * @brief Initialize vector from mesh dimensions
             * @param[in] programCtrl The program control object
             * @param[in] mesh The lattice mesh
             * @return Initialized data vector
             * @throws std::runtime_error if indexed files not found
             **/
            __host__ [[nodiscard]] static const std::vector<T> initialiseVector(const name_t &fileName, const bool empty)
            {
                if (empty)
                {
                    return std::vector<T>(); // Return an empty vector if the file was not found
                }
                else
                {
                    return fileIO::readFieldFile<T>(fileName);
                }
            }

            /**
             * @brief Calculates the destination index for separation and de-interleaving of fields
             * @tparam Deinterleave If true, the fields will be deinterleaved during separation
             * @param[in] mesh The lattice mesh
             * @param[in] localIdx The GPU-order index
             * @param[in] tx,ty,tz Thread coordinates per-block
             * @param[in] bx,by,bz Block coordinates
             * @param[in] GPU_x,GPU_y,GPU_z GPU coordinates
             * @param[in] nxBlocksPerDevice,nyBlocksPerDevice,nzBlocksPerDevice Number of blocks per device in each spatial dimension
             **/
            template <const bool Deinterleave>
            __host__ [[nodiscard]] static host::label_t destIndex(
                const host::latticeMesh &mesh,
                const host::label_t localIdx,
                const host::label_t tx, const host::label_t ty, const host::label_t tz,
                const host::label_t bx, const host::label_t by, const host::label_t bz,
                const host::label_t GPU_x, const host::label_t GPU_y, const host::label_t GPU_z,
                const host::label_t nxBlocksPerDevice,
                const host::label_t nyBlocksPerDevice,
                const host::label_t nzBlocksPerDevice)
            {
                if constexpr (Deinterleave)
                {
                    const host::label_t x = (GPU_x * nxBlocksPerDevice + bx) * block::nx<host::label_t>() + tx;
                    const host::label_t y = (GPU_y * nyBlocksPerDevice + by) * block::ny<host::label_t>() + ty;
                    const host::label_t z = (GPU_z * nzBlocksPerDevice + bz) * block::nz<host::label_t>() + tz;
                    return Cartesian::idx(x, y, z, mesh.dimension<axis::X>(), mesh.dimension<axis::Y>());
                }
                else
                {
                    return localIdx;
                }
            }
        };
    }
}

#endif

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
    A class handling the representation of a complete set of boundary values for
    all fields in a specific region

Namespace
    LBM

SourceFiles
    boundaryRegion.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_BOUNDARYREGION_CUH
#define __MBLBM_BOUNDARYREGION_CUH

namespace LBM
{
    /**
     * @class boundaryRegion
     * @brief Represents a complete set of boundary values for all fields in a specific region
     * @tparam VelocitySet Velocity set configuration defining lattice structure
     *
     * This struct aggregates all field values (density, velocity components, and moments)
     * for a specific boundary region, providing convenient access to individual components.
     **/
    template <class VelocitySet>
    class boundaryRegion
    {
    public:
        // Need to check that the length of fieldNames is 10
        __host__ [[nodiscard]] boundaryRegion(const std::string &regionName)
            : values_{
                  boundaryValue("rho", regionName),
                  boundaryValue("u", regionName),
                  boundaryValue("v", regionName),
                  boundaryValue("w", regionName),
                  boundaryValue("m_xx", regionName),
                  boundaryValue("m_xy", regionName),
                  boundaryValue("m_xz", regionName),
                  boundaryValue("m_yy", regionName),
                  boundaryValue("m_yz", regionName),
                  boundaryValue("m_zz", regionName)}
        {
            if constexpr (verbose())
            {
                print();
            }
        };

        /**
         * @name Field Accessors
         * @brief Provide access to individual field values in the boundary region
         * @return The value of the specified field with appropriate scaling
         **/
        __host__ [[nodiscard]] inline constexpr scalar_t rho() const noexcept
        {
            return values_[index::rho()]();
        }
        __host__ [[nodiscard]] inline constexpr scalar_t u() const noexcept
        {
            return values_[index::u()]();
        }
        __host__ [[nodiscard]] inline constexpr scalar_t v() const noexcept
        {
            return values_[index::v()]();
        }
        __host__ [[nodiscard]] inline constexpr scalar_t w() const noexcept
        {
            return values_[index::w()]();
        }
        __host__ [[nodiscard]] inline constexpr scalar_t m_xx() const noexcept
        {
            return values_[index::xx()]();
        }
        __host__ [[nodiscard]] inline constexpr scalar_t m_xy() const noexcept
        {
            return values_[index::xy()]();
        }
        __host__ [[nodiscard]] inline constexpr scalar_t m_xz() const noexcept
        {
            return values_[index::xz()]();
        }
        __host__ [[nodiscard]] inline constexpr scalar_t m_yy() const noexcept
        {
            return values_[index::yy()]();
        }
        __host__ [[nodiscard]] inline constexpr scalar_t m_yz() const noexcept
        {
            return values_[index::yz()]();
        }
        __host__ [[nodiscard]] inline constexpr scalar_t m_zz() const noexcept
        {
            return values_[index::zz()]();
        }

        /**
         * @brief Print all field values for this boundary region
         * @note Only active when VERBOSE macro is defined
         **/
        void print() const noexcept
        {
            const std::vector<std::string> regionNames({"rho", "u", "v", "w", "m_xx", "m_xy", "m_xz", "m_yy", "m_yz", "m_zz"});
            for (std::size_t field = 0; field < regionNames.size(); field++)
            {
                std::cout << regionNames[field] << ": " << values_[field]() << std::endl;
            }
        }

    private:
        /**
         * @brief Array of boundary values for all fields
         **/
        const boundaryValue<VelocitySet> values_[10];
    };
}

#endif
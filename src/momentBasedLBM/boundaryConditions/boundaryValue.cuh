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
    A class handling the representation of a single boundary value for a
    specific field and region

Namespace
    LBM

SourceFiles
    boundaryValue.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_BOUNDARYVALUE_CUH
#define __MBLBM_BOUNDARYVALUE_CUH

#include "../../momentBasedLBM/velocitySet/velocitySetBase.cuh"

namespace LBM
{
    /**
     * @class boundaryValue
     * @brief Represents a single boundary value for a specific field and region
     * @tparam VelocitySet The velocity set (D3Q19 or D3Q27)
     *
     * This struct reads and stores boundary condition values from configuration files,
     * handling both direct numerical values and equilibrium-based calculations.
     * It automatically applies appropriate scaling based on field type.
     **/
    template <class VelocitySet, const bool Scaled>
    class boundaryValue
    {
    public:
        /**
         * @brief Constructs a boundary value from configuration data
         * @param[in] fieldName Name of the field (e.g., "rho", "U_x", "m_xx")
         * @param[in] regionName Name of the boundary region (e.g., "North", "West")
         * @throws std::runtime_error if field name is invalid or configuration is malformed
         **/
        __host__ [[nodiscard]] boundaryValue(const name_t &fieldName, const name_t &regionName)
            : value(initialiseValue(fieldName, regionName)) {}

        /**
         * @brief Access the stored boundary value
         * @return The boundary value with appropriate scaling applied
         **/
        __host__ [[nodiscard]] inline constexpr scalar_t operator()() const noexcept
        {
            return value;
        }

    private:
        /**
         * @brief The underlying variable
         **/
        const scalar_t value;

        /**
         * @brief Extracts a parameter from the configuration file
         * @tparam safety_check Performs a safety check on the parameter
         * @param[in] fieldName Name of the field to extract
         * @param[in] regionName Name of the boundary region
         * @param[in] initialConditionsName Name of the configuration file (default: "initialConditions")
         * @return The extracted parameter value
         * @throws std::runtime_error if the parameter is not found or is invalid
         * @note This function is used to extract values that MUST be numeric
         **/
        template <const bool safety_check>
        __host__ [[nodiscard]] static scalar_t extractParameter(const name_t &fieldName, const name_t &regionName, const name_t &initialConditionsName)
        {
            const words_t boundaryLines = string::readFile(initialConditionsName);

            // Extracts the entire block of text corresponding to currentField
            const words_t fieldBlock = string::extractBlock(boundaryLines, fieldName, "field");

            // Extracts the block of text corresponding to internalField within the current field block
            const words_t regionFieldBlock = string::extractBlock(fieldBlock, regionName);

            const name_t valueString = string::extractParameterLine(regionFieldBlock, "value");

            if constexpr (safety_check)
            {
                if (string::isNumber(valueString))
                {
                    return string::extractParameter<scalar_t>(regionFieldBlock, "value");
                }
                else
                {
                    throw std::runtime_error("Invalid boundary value " + valueString + ". Value must be a number.");
                    return static_cast<scalar_t>(0);
                }
            }
            else
            {
                return string::extractParameter<scalar_t>(regionFieldBlock, "value");
            }
        }

        /**
         * @brief Initializes the boundary value from configuration file
         * @param[in] fieldName Name of the field to initialize
         * @param[in] regionName Name of the boundary region
         * @param[in] initialConditionsName Name of the configuration file (default: "initialConditions")
         * @return Initialized and scaled boundary value
         * @throws std::runtime_error if configuration is invalid or field name is unrecognized
         *
         * This method reads boundary conditions from a configuration file and handles:
         * - Direct numerical values with appropriate scaling
         * - Equilibrium-based calculations for moment fields
         * - Validation of field names and region names
         **/
        __host__ [[nodiscard]] static scalar_t initialiseValue(const name_t &fieldName, const name_t &regionName, const name_t &initialConditionsName = "initialConditions")
        {
            const words_t boundaryLines = string::readFile(initialConditionsName);

            // Extracts the entire block of text corresponding to currentField
            const words_t fieldBlock = string::extractBlock(boundaryLines, fieldName, "field");

            // Extracts the block of text corresponding to internalField within the current field block
            const words_t regionFieldBlock = string::extractBlock(fieldBlock, regionName);

            // Now read the value line
            const name_t value_ = string::extractParameterLine(regionFieldBlock, "value");

            // Try fixing its value
            if (string::isNumber(value_))
            {
                const std::unordered_set<name_t> allowed = {"rho", "U_x", "U_y", "U_z", "Pi_xx", "Pi_xy", "Pi_xz", "Pi_yy", "Pi_yz", "Pi_zz"};

                const bool isMember = allowed.find(fieldName) != allowed.end();

                if (isMember)
                {
                    // Check to see if it is a moment or a velocity and scale appropriately
                    if (fieldName == "rho")
                    {
                        return string::extractParameter<scalar_t>(regionFieldBlock, "value");
                    }
                    if ((fieldName == "U_x") | (fieldName == "U_y") | (fieldName == "U_z"))
                    {
                        if constexpr (Scaled)
                        {
                            return string::extractParameter<scalar_t>(regionFieldBlock, "value") * velocitySetBase::scale_i<scalar_t>();
                        }
                        else
                        {
                            return string::extractParameter<scalar_t>(regionFieldBlock, "value");
                        }
                    }
                    if ((fieldName == "Pi_xx") | (fieldName == "Pi_yy") | (fieldName == "Pi_zz"))
                    {
                        if constexpr (Scaled)
                        {
                            return string::extractParameter<scalar_t>(regionFieldBlock, "value") * velocitySetBase::scale_ii<scalar_t>();
                        }
                        else
                        {
                            return string::extractParameter<scalar_t>(regionFieldBlock, "value");
                        }
                    }
                    if ((fieldName == "Pi_xy") | (fieldName == "Pi_xz") | (fieldName == "Pi_yz"))
                    {
                        if constexpr (Scaled)
                        {
                            return string::extractParameter<scalar_t>(regionFieldBlock, "value") * velocitySetBase::scale_ij<scalar_t>();
                        }
                        else
                        {
                            return string::extractParameter<scalar_t>(regionFieldBlock, "value");
                        }
                    }
                }

                throw std::runtime_error("Invalid field name \" " + fieldName + "\" for equilibrium distribution");
            }
            // Otherwise, test to see if it is an equilibrium moment
            else if (value_ == "equilibrium")
            {
                // Check to see if the variable is one of the moments
                const std::unordered_set<name_t> allowed = fieldType<6>::makeComponentNames<std::unordered_set<name_t>>("Pi");

                const bool isMember = allowed.find(fieldName) != allowed.end();

                // It is an equilibrium moment
                if (isMember)
                {
                    const char A = fieldName[fieldName.size() - 2];
                    const char B = fieldName[fieldName.size() - 1];

                    const scalar_t u_A = extractParameter<true>("U_" + A, regionName, initialConditionsName);
                    const scalar_t u_B = extractParameter<true>("U_" + B, regionName, initialConditionsName);

                    return velocitySetBase::scale<scalar_t>(A == B) * ((u_A * u_B)) / rho0();
                }
                // Otherwise, not valid
                else
                {
                    std::cerr << "Entry for " << fieldName << " in region " << regionName << " not a valid numerical value and not an equilibrium moment" << std::endl;

                    throw std::runtime_error("Invalid field name for equilibrium distribution");

                    return 0;
                }
            }
            // Not valid
            else
            {
                std::cerr << "Entry for " << fieldName << " in region " << regionName << " not a valid numerical value and not an equilibrium moment" << std::endl;

                throw std::runtime_error("Invalid field name for equilibrium distribution");

                return 0;
            }
        }
    };
}

#endif
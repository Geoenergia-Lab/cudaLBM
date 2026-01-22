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
    A class handling the representation of a single boundary value for a
    specific field and region

Namespace
    LBM

SourceFiles
    boundaryValue.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_BOUNDARYVALUE_CUH
#define __MBLBM_BOUNDARYVALUE_CUH

namespace LBM
{
    /**
     * @class boundaryValue
     * @brief Represents a single boundary value for a specific field and region
     * @tparam VelocitySet Velocity set configuration defining lattice structure
     *
     * This struct reads and stores boundary condition values from configuration files,
     * handling both direct numerical values and equilibrium-based calculations.
     * It automatically applies appropriate scaling based on field type.
     **/
    template <class VelocitySet>
    class boundaryValue
    {
    public:
        /**
         * @brief Constructs a boundary value from configuration data
         * @param[in] fieldName Name of the field (e.g., "rho", "u", "m_xx")
         * @param[in] regionName Name of the boundary region (e.g., "North", "West")
         * @throws std::runtime_error if field name is invalid or configuration is malformed
         **/
        __host__ [[nodiscard]] boundaryValue(const std::string &fieldName, const std::string &regionName)
            : value(initialiseValue(fieldName, regionName)){};

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

        /** *
         * @brief Extracts a parameter from the configuration file
         * @tparam T Type of the parameter to extract
         * @param[in] fieldName Name of the field to extract
         * @param[in] regionName Name of the boundary region
         * @param[in] initialConditionsName Name of the configuration file (default: "initialConditions")
         * @return The extracted parameter value
         * @throws std::runtime_error if the parameter is not found or is invalid
         * @note This function is used to extract values that MUST be numeric
         */
        template <const bool safety_check>
        __host__ [[nodiscard]] static scalar_t extractParameter(const std::string &fieldName, const std::string &regionName, const std::string &initialConditionsName)
        {
            const std::vector<std::string> boundaryLines = string::readFile(initialConditionsName);

            // Extracts the entire block of text corresponding to currentField
            const std::vector<std::string> fieldBlock = string::extractBlock(boundaryLines, fieldName, "field");

            // Extracts the block of text corresponding to internalField within the current field block
            const std::vector<std::string> regionFieldBlock = string::extractBlock(fieldBlock, regionName);

            const std::string valueString = string::extractParameterLine(regionFieldBlock, "value");

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
        __host__ [[nodiscard]] static scalar_t initialiseValue(const std::string &fieldName, const std::string &regionName, const std::string &initialConditionsName = "initialConditions")
        {
            const std::vector<std::string> boundaryLines = string::readFile(initialConditionsName);

            // Extracts the entire block of text corresponding to currentField
            const std::vector<std::string> fieldBlock = string::extractBlock(boundaryLines, fieldName, "field");

            // Extracts the block of text corresponding to internalField within the current field block
            const std::vector<std::string> regionFieldBlock = string::extractBlock(fieldBlock, regionName);

            // Now read the value line
            const std::string value_ = string::extractParameterLine(regionFieldBlock, "value");

            // Try fixing its value
            if (string::isNumber(value_))
            {
                const std::unordered_set<std::string> allowed = {"rho", "u", "v", "w", "m_xx", "m_xy", "m_xz", "m_yy", "m_yz", "m_zz"};

                const bool isMember = allowed.find(fieldName) != allowed.end();

                if (isMember)
                { // Check to see if it is a moment or a velocity and scale appropriately
                    if (fieldName == "rho")
                    {
                        return string::extractParameter<scalar_t>(regionFieldBlock, "value");
                    }
                    if ((fieldName == "u") | (fieldName == "v") | (fieldName == "w"))
                    {
                        return string::extractParameter<scalar_t>(regionFieldBlock, "value") * velocitySet::scale_i<scalar_t>();
                    }
                    if ((fieldName == "m_xx") | (fieldName == "m_yy") | (fieldName == "m_zz"))
                    {
                        return string::extractParameter<scalar_t>(regionFieldBlock, "value") * velocitySet::scale_ii<scalar_t>();
                    }
                    if ((fieldName == "m_xy") | (fieldName == "m_xz") | (fieldName == "m_yz"))
                    {
                        return string::extractParameter<scalar_t>(regionFieldBlock, "value") * velocitySet::scale_ij<scalar_t>();
                    }
                }

                throw std::runtime_error("Invalid field name \" " + fieldName + "\" for equilibrium distribution");
            }
            // Otherwise, test to see if it is an equilibrium moment
            else if (value_ == "equilibrium")
            {
                // Check to see if the variable is one of the moments
                const std::unordered_set<std::string> allowed = {"m_xx", "m_xy", "m_xz", "m_yy", "m_yz", "m_zz"};
                const bool isMember = allowed.find(fieldName) != allowed.end();

                // It is an equilibrium moment
                if (isMember)
                {
                    // Store second-order moments
                    if (fieldName == "m_xx")
                    {
                        const scalar_t u = extractParameter<true>("u", regionName, initialConditionsName);
                        return velocitySet::scale_ii<scalar_t>() * ((u * u)) / rho0<scalar_t>();
                    }
                    else if (fieldName == "m_xy")
                    {
                        const scalar_t u = extractParameter<true>("u", regionName, initialConditionsName);
                        const scalar_t v = extractParameter<true>("v", regionName, initialConditionsName);
                        return velocitySet::scale_ii<scalar_t>() * ((u * v)) / rho0<scalar_t>();
                    }
                    else if (fieldName == "m_xz")
                    {
                        const scalar_t u = extractParameter<true>("u", regionName, initialConditionsName);
                        const scalar_t w = extractParameter<true>("w", regionName, initialConditionsName);
                        return velocitySet::scale_ii<scalar_t>() * ((u * w)) / rho0<scalar_t>();
                    }
                    else if (fieldName == "m_yy")
                    {
                        const scalar_t v = extractParameter<true>("v", regionName, initialConditionsName);
                        return velocitySet::scale_ii<scalar_t>() * ((v * v)) / rho0<scalar_t>();
                    }
                    else if (fieldName == "m_yz")
                    {
                        const scalar_t v = extractParameter<true>("v", regionName, initialConditionsName);
                        const scalar_t w = extractParameter<true>("w", regionName, initialConditionsName);
                        return velocitySet::scale_ii<scalar_t>() * ((v * w)) / rho0<scalar_t>();
                    }
                    else if (fieldName == "m_zz")
                    {
                        const scalar_t w = extractParameter<true>("w", regionName, initialConditionsName);
                        return velocitySet::scale_ii<scalar_t>() * ((w * w)) / rho0<scalar_t>();
                    }
                    return 0; // Should never get here
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
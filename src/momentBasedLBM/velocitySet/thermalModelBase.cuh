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
    Definition of the lattice thermal model; weakly compressible or isothermal

Namespace
    LBM

SourceFiles
    thermalModelBase.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_THERMALMODELBASE_CUH
#define __MBLBM_THERMALMODELBASE_CUH

namespace LBM
{
    /**
     * @brief Enumerated type for indexing pointers to halos
     **/
    typedef enum thermalModelEnum : bool
    {
        Thermal = 0,
        Isothermal = 1
    } thermalModel_t;

    /**
     * @brief Base class for thermal model implementations, providing common functionality and enforcing the thermal model type
     **/
    template <const thermalModel_t ThermalModel>
    class thermalModelBase
    {
    protected:
        /**
         * @brief The calculated value of 1 - c_s^2 * (m_xx + m_yy + m_zz)
         **/
        const scalar_t pics2_;

        /**
         * @brief Calculates 1 - c_s^2 * (m_xx + m_yy + m_zz) for the given diagonal moments
         * @param[in] m_xx The diagonal moment in the x-direction
         * @param[in] m_yy The diagonal moment in the y-direction
         * @param[in] m_zz The diagonal moment in the z-direction
         * @return The calculated value of 1 - c_s^2 * (m_xx + m_yy + m_zz)
         **/
        __device__ __host__ [[nodiscard]] static inline constexpr scalar_t pics2(const scalar_t m_xx, const scalar_t m_yy, const scalar_t m_zz) noexcept
        {
            return static_cast<scalar_t>(1) - velocitySetBase::cs2<scalar_t>() * (m_xx + m_yy + m_zz);
        }

    public:
        static_assert(((ThermalModel == Thermal) || (ThermalModel == Isothermal)), "ThermalModel must be Thermal or Isothermal.");

        /**
         * @brief Constructs a thermal model base with the specified Pics2 value
         * @param[in] val The value of 1 - c_s^2 * (m_xx + m_yy + m_zz) to be stored in the base
         **/
        __device__ __host__ constexpr thermalModelBase(const scalar_t val) noexcept
            : pics2_(val) {}

        /**
         * @brief Returns the value of 1 - c_s^2 * (m_xx + m_yy + m_zz)
         * @return The calculated value of 1 - c_s^2 * (m_xx + m_yy + m_zz)
         **/
        __device__ __host__ [[nodiscard]] inline constexpr scalar_t pics2() const noexcept
        {
            return pics2_;
        }

        /**
         * @brief Returns the type of the thermal model
         * @return The thermal model type (Thermal or Isothermal)
         **/
        __device__ __host__ [[nodiscard]] static inline consteval thermalModel_t modelType() noexcept
        {
            return ThermalModel;
        }
    };

    /**
     * @brief Base template declaration of the thermal model class for specific thermal model types
     * @tparam ThermalModel The thermal model type (Thermal or Isothermal)
     **/
    template <const thermalModel_t ThermalModel>
    class thermalModel : public thermalModelBase<ThermalModel>
    {
    };

    /**
     * @brief Specialization of the thermal model class for the isothermal case, providing specific implementations for isothermal behavior
     **/
    template <>
    class thermalModel<Isothermal> : public thermalModelBase<Isothermal>
    {
        using This = thermalModel<Isothermal>;
        using Base = thermalModelBase<Isothermal>;

    public:
        /**
         * @brief Constructs a thermal model for the isothermal case, calculating the diagonal correction term and Pics2 value based on the provided moments
         * @param[in] moments The calculated moments array (rho, U, Pi)
         **/
        __device__ __host__ [[nodiscard]] thermalModel<Isothermal>(const momentsArray &moments) noexcept
            : Base(Base::pics2(diagonalTerm_[m_i<0>()], diagonalTerm_[m_i<1>()], diagonalTerm_[m_i<2>()])),
              diagonalTerm_(This::diagonalTerm(moments)) {}

        /**
         * @brief Returns the modified diagonal terms for the isothermal formulation
         * @return The modified diagonal terms as a thread::array of size 3
         **/
        __device__ __host__ [[nodiscard]] inline constexpr const thread::array<const scalar_t, 3> &diagonalTerm() const noexcept
        {
            return diagonalTerm_;
        }

        /**
         * @brief Selects between the modified diagonal terms and the original diagonal components based on the moment index
         * @tparam i The moment index
         * @param[in] moments The calculated moments array
         * @return The selected moment value based on the index
         **/
        template <const host::label_t i>
        __device__ __host__ [[nodiscard]] inline constexpr scalar_t moment(const momentsArray &moments) const noexcept
        {
            if constexpr (i == 4)
            {
                return diagonalTerm_[m_i<0>()];
            }
            else if constexpr (i == 7)
            {
                return diagonalTerm_[m_i<1>()];
            }
            else if constexpr (i == 9)
            {
                return diagonalTerm_[m_i<2>()];
            }
            else
            {
                return moments[m_i<i>()];
            }
        }

    private:
        /**
         * @brief Calculates the diagonal correction term for the isothermal velocity set
         * @param[in] moments Moment array (rho, U, Pi)
         **/
        __device__ __host__ [[nodiscard]] static inline constexpr const thread::array<const scalar_t, 3> diagonalTerm(const momentsArray &moments) noexcept
        {
            const scalar_t Delta_m = (moments[q_i<1>()] * moments[q_i<1>()] + moments[q_i<2>()] * moments[q_i<2>()] + moments[q_i<3>()] * moments[q_i<3>()] - moments[q_i<4>()] - moments[q_i<7>()] - moments[q_i<9>()]) / static_cast<scalar_t>(3);

            return {moments[q_i<4>()] + Delta_m, moments[q_i<7>()] + Delta_m, moments[q_i<9>()] + Delta_m};
        }

        /**
         * @brief The modified diagonal terms for the isothermal formulation, calculated from the provided moments
         **/
        const thread::array<const scalar_t, 3> diagonalTerm_;
    };

    template <>
    class thermalModel<Thermal> : public thermalModelBase<Thermal>
    {
        using This = thermalModel<Thermal>;
        using Base = thermalModelBase<Thermal>;

    public:
        /**
         * @brief Constructs a thermal model for the thermal case, calculating the Pics2 value based on the provided moments
         * @param[in] moments The calculated moments array (rho, U, Pi)
         **/
        __device__ __host__ [[nodiscard]] thermalModel<Thermal>(const momentsArray &moments) noexcept
            : Base(Base::pics2(moments[q_i<4>()], moments[q_i<7>()], moments[q_i<9>()])) {}

        /**
         * @brief Selects between the modified diagonal terms and the original diagonal components based on the moment index
         * @tparam i The moment index
         * @param[in] moments The calculated moments array
         * @return The selected moment value based on the index
         **/
        template <const host::label_t i>
        __device__ __host__ [[nodiscard]] inline constexpr scalar_t moment(const momentsArray &moments) const noexcept
        {
            return moments[m_i<i>()];
        }
    };
}

#endif
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
    A class handling the setup of the solver

Namespace
    LBM

SourceFiles
    programControl.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_PROGRAMCONTROL_CUH
#define __MBLBM_PROGRAMCONTROL_CUH

#include "../LBMIncludes.cuh"
#include "../typedefs/typedefs.cuh"
#include "../strings.cuh"
#include "inputControl.cuh"
#include "../fileIO/fileIO.cuh"

namespace LBM
{
    class programControl
    {
    public:
        /**
         * @brief Constructor for the programControl class
         * @param[in] argc First argument passed to main
         * @param[in] argv Second argument passed to main
         **/
        __host__ [[nodiscard]] programControl(const int argc, const char *const argv[]) noexcept
            : input_(inputControl(argc, argv))
        {
            types::assertions::validate<scalar_t>();
            types::assertions::validate<device::label_t>();

            const auto file = string::readFile("programControl");

            caseName_ = string::extractParameter<std::string>(file, "caseName");
            multiphase_ = string::extractParameter<bool>(file, "multiphase");

            // Dual-characteristic auto-detection
            const bool hasUinfA = string::hasParameter(file, "u_inf_A");
            const bool hasUinfB = string::hasParameter(file, "u_inf_B");
            const bool hasLcharA = string::hasParameter(file, "L_char_A");
            const bool hasLcharB = string::hasParameter(file, "L_char_B");

            dualCharacteristic_ = (hasUinfA || hasUinfB || hasLcharA || hasLcharB);

            if (dualCharacteristic_)
            {
                // Require full A/B set
                if (!(hasUinfA && hasUinfB && hasLcharA && hasLcharB))
                {
                    [[maybe_unused]] const errorHandler err(-1, "Dual-characteristic requires all of: u_inf_A, u_inf_B, L_char_A, L_char_B.");
                }

                u_inf_A_ = string::extractParameter<scalar_t>(file, "u_inf_A");
                u_inf_B_ = string::extractParameter<scalar_t>(file, "u_inf_B");
                L_char_A_ = string::extractParameter<scalar_t>(file, "L_char_A");
                L_char_B_ = string::extractParameter<scalar_t>(file, "L_char_B");

                // Legacy single getters map to A
                u_inf_single_ = u_inf_A_;
                L_char_single_ = L_char_A_;
            }
            else
            {
                const bool hasUinf = string::hasParameter(file, "u_inf");
                const bool hasLchar = string::hasParameter(file, "L_char");

                if (!hasUinf || !hasLchar)
                {
                    [[maybe_unused]] const errorHandler err(-1, "Single-characteristic requires both 'u_inf' and 'L_char'.");
                }

                u_inf_single_ = string::extractParameter<scalar_t>(file, "u_inf");
                L_char_single_ = string::extractParameter<scalar_t>(file, "L_char");

                // Map A/B getters to single values
                u_inf_A_ = u_inf_single_;
                u_inf_B_ = u_inf_single_;
                L_char_A_ = L_char_single_;
                L_char_B_ = L_char_single_;
            }

            // Nozzle scaling parameters for SSMD case (default unused)
            const bool hasNozzleScaleA = string::hasParameter(file, "nozzleScale_A");
            const bool hasNozzleScaleB = string::hasParameter(file, "nozzleScale_B");

            if ((hasNozzleScaleA || hasNozzleScaleB) && (!multiphase_ || !dualCharacteristic_))
            {
                [[maybe_unused]] const errorHandler err(-1, "nozzleScale_* only allowed for multiphase dual-characteristic cases (SSMD).");
            }

            if (hasNozzleScaleA != hasNozzleScaleB)
            {
                [[maybe_unused]] const errorHandler err(-1, "Nozzle scaling requires both 'nozzleScale_A' and 'nozzleScale_B' (either specify both or neither).");
            }

            if (hasNozzleScaleA) // implies hasNozzleScaleB
            {
                nozzleScale_A_ = string::extractParameter<scalar_t>(file, "nozzleScale_A");
                nozzleScale_B_ = string::extractParameter<scalar_t>(file, "nozzleScale_B");
            }
            else
            {
                nozzleScale_A_ = static_cast<scalar_t>(0);
                nozzleScale_B_ = static_cast<scalar_t>(0);
            }

            // Reynolds selection
            const bool hasRe = string::hasParameter(file, "Re");
            const bool hasReA = string::hasParameter(file, "Re_A");
            const bool hasReB = string::hasParameter(file, "Re_B");

            if (!multiphase_)
            {
                if (!hasRe)
                {
                    [[maybe_unused]] const errorHandler err(-1, "Single-phase simulation requires 'Re'.");
                }

                Re_ = string::extractParameter<scalar_t>(file, "Re");
                Re_A_ = static_cast<scalar_t>(0);
                Re_B_ = static_cast<scalar_t>(0);

                We_ = static_cast<scalar_t>(0);
                interfaceWidth_ = static_cast<scalar_t>(0);
            }
            else
            {
                if (!(hasReA && hasReB))
                {
                    [[maybe_unused]] const errorHandler err(-1, "Multiphase requires both 'Re_A' and 'Re_B'.");
                }

                Re_ = static_cast<scalar_t>(0);
                Re_A_ = string::extractParameter<scalar_t>(file, "Re_A");
                Re_B_ = string::extractParameter<scalar_t>(file, "Re_B");

                We_ = string::extractParameter<scalar_t>(file, "We");
                interfaceWidth_ = string::extractParameter<scalar_t>(file, "interfaceWidth");
            }

            nTimeSteps_ = string::extractParameter<host::label_t>(file, "nTimeSteps");
            saveInterval_ = string::extractParameter<host::label_t>(file, "saveInterval");
            infoInterval_ = string::extractParameter<host::label_t>(file, "infoInterval");
            latestTime_ = fileIO::latestTime(caseName_);

            // Get the launch time
            const time_t time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

            // Get current working directory
            const std::filesystem::path launchDirectory = std::filesystem::current_path();

            std::cout << "/*---------------------------------------------------------------------------*\\" << std::endl;
            std::cout << "|                                                                             |" << std::endl;
            std::cout << "| cudaLBM: CUDA-based moment representation Lattice Boltzmann Method          |" << std::endl;
            std::cout << "| Developed at UDESC - State University of Santa Catarina                     |" << std::endl;
            std::cout << "| Website: https://www.udesc.br                                               |" << std::endl;
            std::cout << "| Github: https://github.com/geoenergiaUDESC/cudaLBM                          |" << std::endl;
            std::cout << "|                                                                             |" << std::endl;
            std::cout << "\\*---------------------------------------------------------------------------*/" << std::endl;
            std::cout << std::endl;
            std::cout << "programControl:" << std::endl;
            std::cout << "{" << std::endl;
            std::cout << "    programName: " << input_.commandLine()[0] << ";" << std::endl;
            std::cout << "    launchTime: " << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S") << ";" << std::endl;
            std::cout << "    launchDirectory: " << launchDirectory.string() << ";" << std::endl;
            if (deviceList().size() > 0)
            {
                std::cout << "    deviceList: [";

                for (host::label_t i = 0; i < deviceList().size() - 1; i++)
                {
                    std::cout << deviceList()[i] << ", ";
                }
            }
            std::cout << "    caseName: " << caseName_ << ";" << std::endl;
            std::cout << "    multiphase: " << (multiphase_ ? "true" : "false") << ";" << std::endl;
            std::cout << "    dualCharacteristic: " << (dualCharacteristic_ ? "true" : "false") << ";" << std::endl;
            if (!multiphase_)
            {
                std::cout << "    Re = " << Re_ << ";" << std::endl;
            }
            else
            {
                std::cout << "    Re_A = " << Re_A_ << ";" << std::endl;
                std::cout << "    Re_B = " << Re_B_ << ";" << std::endl;
                std::cout << "    We = " << We_ << ";" << std::endl;
                std::cout << "    interfaceWidth = " << interfaceWidth_ << ";" << std::endl;
            }
            if (!dualCharacteristic_)
            {
                std::cout << "    u_inf = " << u_inf_single_ << ";" << std::endl;
                std::cout << "    L_char = " << L_char_single_ << ";" << std::endl;
            }
            else
            {
                std::cout << "    u_inf_A = " << u_inf_A_ << ";" << std::endl;
                std::cout << "    u_inf_B = " << u_inf_B_ << ";" << std::endl;
                std::cout << "    L_char_A = " << L_char_A_ << ";" << std::endl;
                std::cout << "    L_char_B = " << L_char_B_ << ";" << std::endl;
            }
            if (hasNozzleScaleA)
            {
                std::cout << "    nozzleScale_A = " << nozzleScale_A_ << ";" << std::endl;
                std::cout << "    nozzleScale_B = " << nozzleScale_B_ << ";" << std::endl;
            }
            std::cout << "    nTimeSteps = " << nTimeSteps_ << ";" << std::endl;
            std::cout << "    saveInterval = " << saveInterval_ << ";" << std::endl;
            std::cout << "    infoInterval = " << infoInterval_ << ";" << std::endl;
            std::cout << "    latestTime = " << latestTime_ << ";" << std::endl;
            std::cout << "    scalarSize: " << sizeof(scalar_t) * 8 << ";" << std::endl;
            std::cout << "    labelType: uint" << sizeof(device::label_t) * 8 << "_t" << ";" << std::endl;
            std::cout << "};" << std::endl;
            std::cout << std::endl;

            if (deviceList().size() > 0)
            {
                for (host::label_t virtualDeviceIndex = 0; virtualDeviceIndex < deviceList().size(); virtualDeviceIndex++)
                {
                    errorHandler::check(cudaSetDevice(deviceList()[virtualDeviceIndex]));

                    if (!isMultiphase())
                    {
                        // Allocate symbols on the GPU
                        const scalar_t viscosityTemp = u_inf() * L_char() / Re();
                        const scalar_t tauTemp = static_cast<scalar_t>(0.5) + static_cast<scalar_t>(3.0) * viscosityTemp;
                        const scalar_t omegaTemp = static_cast<scalar_t>(1.0) / tauTemp;
                        const scalar_t t_omegaVarTemp = static_cast<scalar_t>(1) - omegaTemp;
                        const scalar_t omegaVar_d2Temp = omegaTemp * static_cast<scalar_t>(0.5);

                        device::copyToSymbol(device::L_char, L_char());
                        device::copyToSymbol(device::Re, Re());
                        device::copyToSymbol(device::tau, tauTemp);
                        device::copyToSymbol(device::omega, omegaTemp);
                        device::copyToSymbol(device::t_omegaVar, t_omegaVarTemp);
                        device::copyToSymbol(device::omegaVar_d2, omegaVar_d2Temp);
                    }
                    else
                    {
                        device::copyToSymbol(device::nozzleScale_A, nozzleScale_A());
                        device::copyToSymbol(device::nozzleScale_B, nozzleScale_B());

                        scalar_t tauTempA;
                        scalar_t tauTempB;

                        if (dualCharacteristic())
                        {
                            const scalar_t viscosityTempA = u_inf_A() * L_char_A() / Re_A();
                            const scalar_t viscosityTempB = u_inf_B() * L_char_B() / Re_B();
                            tauTempA = static_cast<scalar_t>(0.5) + static_cast<scalar_t>(3.0) * viscosityTempA;
                            tauTempB = static_cast<scalar_t>(0.5) + static_cast<scalar_t>(3.0) * viscosityTempB;
                            const scalar_t sigmaTemp = (u_inf_B() * u_inf_B() * L_char_B()) / We();

                            device::copyToSymbol(device::L_char_A, L_char_A());
                            device::copyToSymbol(device::L_char_B, L_char_B());
                            device::copyToSymbol(device::sigma, sigmaTemp);
                        }
                        else
                        {
                            const scalar_t viscosityTempA = u_inf() * L_char() / Re_A();
                            const scalar_t viscosityTempB = u_inf() * L_char() / Re_B();
                            tauTempA = static_cast<scalar_t>(0.5) + static_cast<scalar_t>(3.0) * viscosityTempA;
                            tauTempB = static_cast<scalar_t>(0.5) + static_cast<scalar_t>(3.0) * viscosityTempB;
                            const scalar_t sigmaTemp = (u_inf() * u_inf() * L_char()) / We();

                            device::copyToSymbol(device::L_char, L_char());
                            device::copyToSymbol(device::sigma, sigmaTemp);
                        }

                        const scalar_t omegaTemp = static_cast<scalar_t>(1.0) / tauTempA;
                        const scalar_t gammaTemp = static_cast<scalar_t>(2.0) / interfaceWidth();

                        device::copyToSymbol(device::tau, tauTempA); // For compatibility. Ctrl+Shift+F -> CHECKPOINT to find place of future fix
                        device::copyToSymbol(device::tau_A, tauTempA);
                        device::copyToSymbol(device::tau_B, tauTempB);
                        device::copyToSymbol(device::omega, omegaTemp);
                        device::copyToSymbol(device::gamma, gammaTemp);
                    }
                }
            }

            // Make sure we synchronize and set active device to 0
            // Probably unnecessary but nice to do it anyway
            if (deviceList().size() > 0)
            {
                errorHandler::check(cudaDeviceSynchronize());
                errorHandler::check(cudaSetDevice(deviceList()[0]));
                errorHandler::check(cudaDeviceSynchronize());
            }
        };

        /**
         * @brief Destructor for the programControl class
         **/
        ~programControl() noexcept
        {
            std::cout << std::endl;
            std::cout << "End" << std::endl;
            std::cout << std::endl;
        }

        /**
         * @brief Disable copying
         **/
        __host__ [[nodiscard]] programControl(const programControl &) = delete;
        __host__ [[nodiscard]] programControl &operator=(const programControl &) = delete;

        /**
         * @brief Returns the name of the case
         * @return A const name_t
         **/
        __host__ [[nodiscard]] inline constexpr const name_t &caseName() const noexcept
        {
            return caseName_;
        }

        /**
         * @brief Returns multiphase or not
         * @return Multiphase bool
         **/
        __device__ __host__ [[nodiscard]] inline constexpr bool isMultiphase() const noexcept
        {
            return multiphase_;
        }

        /**
         * @brief CHECKPOINT
         * @return CHECKPOINT
         **/
        __device__ __host__ [[nodiscard]] inline constexpr bool dualCharacteristic() const noexcept
        {
            return dualCharacteristic_;
        }

        /**
         * @brief Returns the array of device indices
         * @return A read-only reference to deviceList_ contained within input_
         **/
        __host__ [[nodiscard]] inline constexpr const std::vector<deviceIndex_t> &deviceList() const noexcept
        {
            return input_.deviceList();
        }

        /**
         * @brief Returns the Reynolds number
         * @return The Reynolds number
         **/
        __device__ __host__ [[nodiscard]] inline constexpr scalar_t Re() const noexcept
        {
            return Re_;
        }

        /**
         * @brief Returns the fluid A Reynolds number
         * @return The Reynolds number for fluid A
         **/
        __device__ __host__ [[nodiscard]] inline constexpr scalar_t Re_A() const noexcept
        {
            return Re_A_;
        }

        /**
         * @brief Returns the fluid B Reynolds number
         * @return The Reynolds number for fluid B
         **/
        __device__ __host__ [[nodiscard]] inline constexpr scalar_t Re_B() const noexcept
        {
            return Re_B_;
        }

        /**
         * @brief Returns the Weber number
         * @return The Weber number
         **/
        __device__ __host__ [[nodiscard]] inline constexpr scalar_t We() const noexcept
        {
            return We_;
        }

        /**
         * @brief Returns the interface width
         * @return The interface width
         **/
        __device__ __host__ [[nodiscard]] inline constexpr scalar_t interfaceWidth() const noexcept
        {
            return interfaceWidth_;
        }

        /**
         * @brief Returns the characteristic velocity
         * @return The characteristic velocity
         **/
        __device__ __host__ [[nodiscard]] inline constexpr scalar_t u_inf() const noexcept
        {
            return dualCharacteristic_ ? u_inf_A_ : u_inf_single_;
        }

        /**
         * @brief Returns the characteristic length
         * @return The characteristic length
         **/
        __device__ __host__ [[nodiscard]] inline constexpr scalar_t L_char() const noexcept
        {
            return dualCharacteristic_ ? L_char_A_ : L_char_single_;
        }

        /**
         * @brief Returns the characteristic velocity of fluid A
         * @return The characteristic velocity of fluid A
         **/
        __device__ __host__ [[nodiscard]] inline constexpr scalar_t u_inf_A() const noexcept
        {
            return u_inf_A_;
        }

        /**
         * @brief Returns the characteristic velocity of fluid B
         * @return The characteristic velocity of fluid B
         **/
        __device__ __host__ [[nodiscard]] inline constexpr scalar_t u_inf_B() const noexcept
        {
            return u_inf_B_;
        }

        /**
         * @brief Returns the nozzle scaling factor of fluid A
         * @return The nozzle scaling factor of fluid A
         **/
        __device__ __host__ [[nodiscard]] inline constexpr scalar_t nozzleScale_A() const noexcept
        {
            return nozzleScale_A_;
        }

        /**
         * @brief Returns the nozzle scaling factor of fluid B
         * @return The nozzle scaling factor of fluid B
         **/
        __device__ __host__ [[nodiscard]] inline constexpr scalar_t nozzleScale_B() const noexcept
        {
            return nozzleScale_B_;
        }

        /**
         * @brief Returns the characteristic length of fluid A
         * @return The characteristic length of fluid A
         **/
        __device__ __host__ [[nodiscard]] inline constexpr scalar_t L_char_A() const noexcept
        {
            return L_char_A_;
        }

        /**
         * @brief Returns the characteristic length of fluid B
         * @return The characteristic length of fluid B
         **/
        __device__ __host__ [[nodiscard]] inline constexpr scalar_t L_char_B() const noexcept
        {
            return L_char_B_;
        }

        /**
         * @brief Returns the total number of simulation time steps
         * @return The total number of simulation time steps
         **/
        __device__ __host__ [[nodiscard]] inline constexpr host::label_t nt() const noexcept
        {
            return nTimeSteps_;
        }

        /**
         * @brief Decide whether or not the program should perform a checkpoint
         * @return True if the program should checkpoint, false otherwise
         **/
        __device__ __host__ [[nodiscard]] inline constexpr bool save(const host::label_t timeStep) const noexcept
        {
            return (timeStep % saveInterval_) == 0;
        }

        /**
         * @brief Decide whether or not the program should perform a checkpoint
         * @return True if the program should checkpoint, false otherwise
         **/
        __device__ __host__ [[nodiscard]] inline constexpr bool print(const host::label_t timeStep) const noexcept
        {
            return (timeStep % infoInterval_) == 0;
        }

        /**
         * @brief Returns the latest time step of the solution files contained within the current directory
         * @return The latest time step as a host::label_t
         **/
        __device__ __host__ [[nodiscard]] inline constexpr host::label_t latestTime() const noexcept
        {
            return latestTime_;
        }

        /**
         * @brief Provides read-only access to the input control
         * @return A const reference to an inputControl object
         **/
        __host__ [[nodiscard]] inline constexpr const inputControl &input() const noexcept
        {
            return input_;
        }

        /**
         * @brief Veriefies if the command line has the argument -type
         * @return A string representing the convertion type passed at the command line
         * @param[in] programCtrl The program control object
         **/
        __host__ [[nodiscard]] const name_t getArgument(const name_t &argument) const
        {
            if (input_.isArgPresent(argument))
            {
                for (host::label_t arg = 0; arg < commandLine().size(); arg++)
                {
                    if (commandLine()[arg] == argument)
                    {
                        if (arg + 1 == commandLine().size())
                        {
                            throw std::runtime_error("Argument " + argument + " not specified: the correct syntax is " + argument + " Arg");
                        }
                        else
                        {
                            return commandLine()[arg + 1];
                        }
                    }
                }
            }

            throw std::runtime_error("Argument " + argument + " not specified: the correct syntax is " + argument + " Arg");
        }

        /**
         * @brief Provides read-only access to the arguments supplied at the command line
         * @return The command line input as a vector of strings
         **/
        __host__ [[nodiscard]] inline constexpr const words_t &commandLine() const noexcept
        {
            return input_.commandLine();
        }

        /**
         * @brief Configures a kernel function to prefer shared memory and sets its dynamic shared memory size
         * @tparam smem_alloc_size The amount of shared memory (in bytes) to allocate for the kernel
         * @tparam T The function type (e.g., a lambda or a function pointer)
         * @param[in] func The kernel function to configure
         **/
        template <const host::label_t smem_alloc_size, class T>
        __host__ void configure(T *func) const
        {
            for (host::label_t VirtualDeviceIndex = 0; VirtualDeviceIndex < deviceList().size(); VirtualDeviceIndex++)
            {
                errorHandler::check(cudaDeviceSynchronize());
                errorHandler::check(cudaSetDevice(deviceList()[VirtualDeviceIndex]));
                errorHandler::check(cudaDeviceSynchronize());
                errorHandler::check(cudaFuncSetCacheConfig(func, cudaFuncCachePreferShared));
                errorHandler::check(cudaDeviceSynchronize());
                errorHandler::check(cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_alloc_size));
                errorHandler::check(cudaDeviceSynchronize());
            }
        }

    private:
        /**
         * @brief A reference to the input control object
         **/
        const inputControl input_;

        /**
         * @brief The name of the simulation case
         **/
        name_t caseName_;

        /**
         * @brief Whether the simulation is multiphase
         **/
        bool multiphase_;

        /**
         * @brief Whether the simulation uses separate characteristic scales for fluids A and B
         **/
        bool dualCharacteristic_;

        /**
         * @brief The Reynolds number
         **/
        scalar_t Re_;

        /**
         * @brief The fluid A Reynolds number
         **/
        scalar_t Re_A_;

        /**
         * @brief The fluid B Reynolds number
         **/
        scalar_t Re_B_;

        /**
         * @brief The Weber number
         **/
        scalar_t We_;

        /**
         * @brief The interface width
         **/
        scalar_t interfaceWidth_;

        /**
         * @brief The characteristic velocity
         **/
        scalar_t u_inf_single_;

        /**
         * @brief The characteristic length scale
         **/
        scalar_t L_char_single_;

        /**
         * @brief The characteristic velocity of fluid A
         **/
        scalar_t u_inf_A_;

        /**
         * @brief The characteristic velocity of fluid B
         **/
        scalar_t u_inf_B_;

        /**
         * @brief The characteristic length scale of fluid A
         **/
        scalar_t L_char_A_;

        /**
         * @brief The characteristic length scale of fluid B
         **/
        scalar_t L_char_B_;

        /**
         * @brief The nozzle scaling factor of fluid A
         **/
        scalar_t nozzleScale_A_;

        /**
         * @brief The nozzle scaling factor of fluid B
         **/
        scalar_t nozzleScale_B_;

        /**
         * @brief Total number of simulation time steps, the save interval, info output interval and the latest time step at program start
         **/
        host::label_t nTimeSteps_;
        host::label_t saveInterval_;
        host::label_t infoInterval_;
        host::label_t latestTime_;

        /**
         * @brief Reads a variable from the caseInfo file into a parameter of type T
         * @return The variable as type T
         * @param[in] varName The name of the variable to read
         **/
        template <typename T>
        __host__ [[nodiscard]] T initialiseConst(const name_t &varName) const noexcept
        {
            return string::extractParameter<T>(string::readFile("programControl"), varName);
        }
    };
}

#include "streamHandler.cuh"

#endif
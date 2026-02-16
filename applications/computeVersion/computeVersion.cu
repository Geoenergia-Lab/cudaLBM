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
    Executable to detect CUDA-capable hardware and generate a hardware info file

Namespace
    LBM

SourceFiles
    computeVersion.cu

\*---------------------------------------------------------------------------*/

#include "computeVersion.cuh"

using namespace LBM;

int main(const int argc, const char *const argv[])
{
    const inputControl input(argc, argv);

    // If we supply the -countDevices argument, just do the device count
    if (input.isArgPresent("-countDevices"))
    {
        const deviceIndex_t deviceCount = countDevices<false>();

        std::cout << "(";
        if (deviceCount > 1)
        {
            for (deviceIndex_t index = 0; index < deviceCount - 1; index++)
            {
                std::cout << index << " ";
            }
            std::cout << deviceCount - 1;
        }
        else
        {
            std::cout << 0;
        }
        std::cout << ")" << std::endl;

        return 0;
    }

    const name_t CUDALBM_ARCHITECTURE_DETECTION = getEnvironmentVariable("CUDALBM_ARCHITECTURE_DETECTION", "Automatic");
    const name_t CUDALBM_ARCHITECTURE_VERSION = getEnvironmentVariable("CUDALBM_ARCHITECTURE_VERSION");
    const name_t CUDALBM_BUILD_DIR = getEnvironmentVariable("CUDALBM_BUILD_DIR");
    const name_t CUDALBM_BIN_DIR = getEnvironmentVariable("CUDALBM_BIN_DIR");
    const name_t CUDALBM_INCLUDE_DIR = getEnvironmentVariable("CUDALBM_INCLUDE_DIR");

    // Now handle the file path setup
    // If the path doesn't exist, create it
    if (!std::filesystem::exists(CUDALBM_BUILD_DIR))
    {
        std::filesystem::create_directory(CUDALBM_BUILD_DIR);
    }
    if (!std::filesystem::exists(CUDALBM_INCLUDE_DIR))
    {
        std::filesystem::create_directory(CUDALBM_INCLUDE_DIR);
    }
    if (!std::filesystem::exists(CUDALBM_BIN_DIR))
    {
        std::filesystem::create_directory(CUDALBM_BIN_DIR);
    }

    const std::filesystem::path outputFilePath = CUDALBM_INCLUDE_DIR + "/hardware.info";
    std::ofstream outputFile(outputFilePath);

    const time_t time_now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

    outputFile << "# Hardware information file automatically generated" << std::endl;
    outputFile << "# Compiled on: " << compileTimestamp() << std::endl;
    outputFile << "# Executed on: " << std::put_time(std::localtime(&time_now), "%Y-%m-%d %H:%M:%S") << std::endl;
    outputFile << std::endl;

    if (CUDALBM_ARCHITECTURE_DETECTION == "Manual")
    {
        const name_t all_arch_flags = "-gencode arch=compute_" + CUDALBM_ARCHITECTURE_VERSION + ",code=sm_" + CUDALBM_ARCHITECTURE_VERSION;
        const name_t all_lto_flags = "-gencode arch=compute_" + CUDALBM_ARCHITECTURE_VERSION + ",code=lto_" + CUDALBM_ARCHITECTURE_VERSION;
        outputFile << "# Consolidated architecture flags for all found GPUs (no duplicates)" << std::endl;
        outputFile << "NVCXX_ALL_ARCHFLAGS = " << all_arch_flags << " " << all_lto_flags << std::endl;
    }
    else
    {
        const deviceIndex_t deviceCount = countDevices<false>();

        name_t all_arch_flags = "";

        for (deviceIndex_t i = 0; i < deviceCount; ++i)
        {
            const cudaDeviceProp props = getDeviceProperties(i);

            const name_t current_arch_flag =
                "-gencode arch=compute_" +
                std::to_string(props.major) +
                std::to_string(props.minor) +
                ",code=sm_" +
                std::to_string(props.major) +
                std::to_string(props.minor);

            const name_t current_lto_flag =
                " -gencode arch=compute_" +
                std::to_string(props.major) +
                std::to_string(props.minor) +
                ",code=lto_" +
                std::to_string(props.major) +
                std::to_string(props.minor);

            if (all_arch_flags.find(current_arch_flag) == name_t::npos)
            {
                all_arch_flags += current_arch_flag;
            }

            if (all_arch_flags.find(current_lto_flag) == name_t::npos)
            {
                all_arch_flags += current_lto_flag;
            }

            outputFile << "# Properties for CUDA Device ID: " << i << std::endl;
            outputFile << "GPU_NAME_" << i << " = " << props.name << std::endl;
            outputFile << "GPU_ARCH_MAJOR_" << i << " = " << props.major << std::endl;
            outputFile << "GPU_ARCH_MINOR_" << i << " = " << props.minor << std::endl;
            outputFile << "GPU_GLOBAL_MEM_MB_" << i << " = " << props.totalGlobalMem / (1024 * 1024) << std::endl;
            outputFile << "GPU_SHARED_MEM_PER_BLOCK_KB_" << i << " = " << props.sharedMemPerBlock / 1024 << std::endl;
            outputFile << "GPU_REGS_PER_BLOCK_" << i << " = " << props.regsPerBlock << std::endl;
            outputFile << "GPU_MAX_THREADS_PER_BLOCK_" << i << " = " << props.maxThreadsPerBlock << std::endl;
            outputFile << "GPU_MULTIPROCESSOR_COUNT_" << i << " = " << props.multiProcessorCount << std::endl;
            outputFile << std::endl;

            outputFile << "# Consolidated architecture flags for all found GPUs (no duplicates)" << std::endl;
            outputFile << "NVCXX_ALL_ARCHFLAGS = " << all_arch_flags << std::endl;
        }
    }

    outputFile.close();

    return 0;
}
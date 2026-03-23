# Top-level Makefile

# Check if required environment variables are set
ifeq ($(CUDALBM_BUILD_DIR),)
$(error CUDALBM_BUILD_DIR is not set. Please run "source bashrc" in the project directory first)
endif

ifeq ($(CUDALBM_BIN_DIR),)
$(error CUDALBM_BIN_DIR is not set. Please run "source bashrc" in the project directory first)
endif

ifeq ($(CUDALBM_INCLUDE_DIR),)
$(error CUDALBM_INCLUDE_DIR is not set. Please run "source bashrc" in the project directory first)
endif

ifeq ($(CUDALBM_CUDA_DIR),)
$(error CUDALBM_CUDA_DIR is not set. Please run "source bashrc" in the project directory first)
endif

TOOL_SUBDIRS = applications/computeVersion
GPU_SUBDIRS = applications/solvers/momentBasedD3Q19 applications/solvers/momentBasedD3Q27 applications/solvers/isothermalD3Q19 applications/solvers/isothermalD3Q27 applications/postProcessing/fieldConvert applications/postProcessing/fieldCalculate
SUBDIRS = $(TOOL_SUBDIRS) $(GPU_SUBDIRS)

.PHONY: all clean install uninstall $(SUBDIRS) directories

all: directories $(SUBDIRS)

# Create build directories
directories:
	@ mkdir -p $(CUDALBM_BUILD_DIR)
	@ mkdir -p $(CUDALBM_BIN_DIR)
	@ mkdir -p $(CUDALBM_INCLUDE_DIR)

# Compile and run computeVersion to generate hardware.info
$(CUDALBM_INCLUDE_DIR)/hardware.info: directories
	@ $(MAKE) -C applications/computeVersion install
	@ computeVersion

# Compile computeVersion (tool subdirectories)
$(TOOL_SUBDIRS): directories
	$(MAKE) -C $@

# Compile GPU applications only after hardware.info is generated
$(GPU_SUBDIRS): $(CUDALBM_INCLUDE_DIR)/hardware.info
	$(MAKE) -C $@

# Clean all projects
clean:
	@ for dir in $(SUBDIRS); do $(MAKE) -C $$dir clean; done
	@ rm -rf $(CUDALBM_BUILD_DIR)

# Install all projects
install: directories $(CUDALBM_INCLUDE_DIR)/hardware.info
	@ for dir in $(SUBDIRS); do $(MAKE) -C $$dir install; done

# Uninstall all projects
uninstall:
	@ for dir in $(SUBDIRS); do $(MAKE) -C $$dir uninstall; done
	@ rm -rf $(CUDALBM_BIN_DIR) $(CUDALBM_INCLUDE_DIR)
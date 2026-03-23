default: clean $(EXECUTABLE)

$(EXECUTABLE):
	$(NVCXX) $(NVCXX_FLAGS) $(SOURCE) -o $@

install: clean uninstall $(EXECUTABLE)
	@ mkdir -p $(CUDALBM_BIN_DIR)
	@ cp $(EXECUTABLE) $(CUDALBM_BIN_DIR)/
	@ rm -f $(EXECUTABLE)

clean:
	@ rm -f $(EXECUTABLE)

uninstall:
	@ rm -f $(CUDALBM_BIN_DIR)/$(EXECUTABLE)
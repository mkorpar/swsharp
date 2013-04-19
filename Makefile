# swsharp must be first on the list
# uncomment swsharpdbmpi module if mpi is available 
MODULES = swsharp swsharpn swsharpp swsharpnc # swsharpdb swsharpout swsharpdbmpi

INC_DIR = include
LIB_DIR = lib
EXC_DIR = bin

all: TARGETS=install
debug: TARGETS=debug install
win: TARGETS=win
clean: TARGETS=remove clean

all: $(MODULES)

debug: $(MODULES)

win: $(MODULES)

clean: $(MODULES)
	@echo [RM] removing
	@rm $(INC_DIR) $(LIB_DIR) $(EXC_DIR) -rf

$(MODULES):
	@echo [MOD] $@
	@$(MAKE) -s -C $@ $(TARGETS)

.PHONY: $(MODULES)

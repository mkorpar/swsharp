INC_DIR = include
LIB_DIR = lib
EXC_DIR = bin

# uncomment swsharpdbmpi module if mpi is available 
CORE = swsharp
MODULES = swsharpn swsharpp swsharpnc swsharpdb swsharpout swsharpdbmpi

all: TARGETS=install
debug: TARGETS=debug install
win: TARGETS=win
clean: TARGETS=remove clean

all: $(CORE) $(MODULES)

debug: $(CORE) $(MODULES)

win: $(CORE) $(MODULES)

clean: $(CORE) $(MODULES)
	@echo [RM] removing
	@rm $(INC_DIR) $(LIB_DIR) $(EXC_DIR) -rf

$(CORE): 
	@echo [CORE] $@
	@$(MAKE) -s -C $@ $(TARGETS)

$(MODULES): $(CORE)
	@echo [MOD] $@
	@$(MAKE) -s -C $@ $(TARGETS)

.PHONY: $(CORE) $(MODULES)

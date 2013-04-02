CC = gcc
CP = g++
CU = nvcc
LD = nvcc
DX = doxygen

NAME = swsharpp

OBJ_DIR = obj
SRC_DIR = src
DOC_DIR = doc
INC_DIR = ../include/$(NAME)
LIB_DIR = ../lib
EXC_DIR = ../bin
WIN_DIR = ../swsharpwin/$(NAME)

I_CMD = $(addprefix -I, $(SRC_DIR) ../include )
L_CMD = $(addprefix -L, ../lib )

DEP_LIBS = ../lib/libswsharp.a

CC_FLAGS = $(I_CMD) -O3 -Wall
CP_FLAGS = $(CC_FLAGS)
CU_FLAGS = $(I_CMD) -O3 -arch sm_13
LD_FLAGS = $(I_CMD) $(L_CMD) -lswsharp -lpthread

SRC = $(shell find $(SRC_DIR) -type f -regex ".*\.\(cu\|c\|cpp\)")
OBJ = $(subst $(SRC_DIR), $(OBJ_DIR), $(addsuffix .o, $(basename $(SRC))))
DEP = $(OBJ:.o=.d)
API = $(shell find $(SRC_DIR) -type f -regex ".*.h")
INC = $(subst $(SRC_DIR), $(INC_DIR), $(API))
LIB = $(LIB_DIR)/lib$(NAME).a
EXC = $(NAME)
BIN = $(EXC_DIR)/$(EXC)
DOC = $(DOC_DIR)/Doxyfile
WIN = $(subst $(SRC_DIR), $(WIN_DIR), $(API) $(SRC))

debug: CC_FLAGS := $(CC_FLAGS) -DDEBUG -DTIMERS
debug: CP_FLAGS := $(CP_FLAGS) -DDEBUG -DTIMERS
debug: CU_FLAGS := $(CU_FLAGS) -DDEBUG -DTIMERS --ptxas-options=-v

all: $(EXC)
debug: all

install: bin win

bin: $(BIN)

include: $(INC)

lib: $(LIB)

win: $(WIN)

$(EXC): $(OBJ) $(DEP_LIBS)
	@echo [LD] $@
	@mkdir -p $(dir $@)
	@$(LD) $(OBJ) -o $@ $(LD_FLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@echo [CC] $<
	@mkdir -p $(dir $@)
	@$(CC) $< -c -o $@ -MMD $(CC_FLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@echo [CP] $<
	@mkdir -p $(dir $@)
	@$(CP) $< -c -o $@ -MMD $(CP_FLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@echo [CU] $<
	@mkdir -p $(dir $@)
	@$(CU) $< -M -o $(@:.o=.d) $(CU_FLAGS) --output-directory $(dir $@)
	@$(CU) $< -c -o $@ $(CU_FLAGS)

$(INC_DIR)/%.h: $(SRC_DIR)/%.h
	@echo [CP] $@
	@mkdir -p $(dir $@)
	@cp $< $@
	
$(LIB): $(OBJ)
	@echo [AR] $@
	@mkdir -p $(dir $@)
	@ar rcs $(LIB) $(OBJ)

$(BIN): $(EXC)
	@echo [CP] $@
	@mkdir -p $(dir $@)
	@cp $< $@

$(WIN_DIR)/%: $(SRC_DIR)/%
	@echo [CP] $@
	@mkdir -p $(dir $@)
	@cp $< $@

docs:
	@echo [DX] generating documentation
	@$(DX) $(DOC)
	
clean:
	@echo [RM] cleaning
	@rm $(OBJ_DIR) $(EXC) -rf

remove:
	@echo [RM] removing
	@rm $(INC_DIR) $(LIB) $(BIN) $(EXC) $(WIN) -rf

-include $(DEP)
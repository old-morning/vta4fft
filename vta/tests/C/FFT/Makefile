XCC ?= $(shell which gcc)
XPP ?= $(shell which g++)
XAR ?= $(shell which gcc-ar)

CFLAGS += -O2 -Wall -std=c++17 -c
CFLAGS += -DDMLC_USE_LOGGING_LIBRARY="<tvm/runtime/logging.h>"
CFLAGS += -I$(TVM_PATH)
CFLAGS += -I$(TVM_PATH)/include
CFLAGS += -isystem $(TVM_PATH)/3rdparty/dlpack/include
CFLAGS += -isystem $(TVM_PATH)/3rdparty/dmlc-core/include
CFLAGS += -I$(VTA_HW_PATH)/include


run-fsim: main-fsim.cc
	$(XCC) $(CFLAGS) -o main-fsim.o main-fsim.cc
	$(XCC) -o main-fsim main-fsim.o -L$(TVM_PATH)/build -lvta_fsim -ltvm_runtime
	./main-fsim

run-tsim: main-tsim.cc
	$(XPP) $(CFLAGS) -o main-tsim.o main-tsim.cc
	$(XPP) -o main-tsim main-tsim.o -L$(TVM_PATH)/build -ltvm_runtime -lvta_tsim
	./main-tsim

clean:
	rm -rf main-fsim.o main-tsim.o main-fsim main-tsim
	

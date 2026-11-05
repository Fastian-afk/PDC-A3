# Imaad Fazal | 23i-0656 | Section C | Assignment 3
MPICC ?= mpicc
CPPFLAGS ?=
CFLAGS ?= -O3 -fopenmp -Wall -Wextra -Wpedantic
LDFLAGS ?=
LDLIBS ?= -lm
PYTHON ?= python3

Q1_NAMES := q1-serial-ijk q1-serial-ikj q1-omp q1-a q1-b q1-c
Q2_NAMES := q2-omp q2-a q2-b q2-balanced
Q3_NAMES := q3-reduce-bcast q3-allreduce q3-ring
PROGRAMS := $(addprefix bin/,$(Q1_NAMES) $(Q2_NAMES) $(Q3_NAMES))

.PHONY: all clean check plan package
all: $(PROGRAMS)

bin:
	mkdir -p $@

define BUILD_Q1
bin/$(word $(1),$(Q1_NAMES)): Q1/23i-0656-C-TASK1-V$(1).c Q1/23i-0656-C-TASK1-common.h common/pdc.h | bin
	$$(MPICC) $$(CPPFLAGS) $$(CFLAGS) $$< $$(LDFLAGS) $$(LDLIBS) -o $$@
endef
$(foreach v,1 2 3 4 5 6,$(eval $(call BUILD_Q1,$(v))))

define BUILD_Q2
bin/$(word $(1),$(Q2_NAMES)): Q2/23i-0656-C-TASK2-V$(1).c Q2/23i-0656-C-TASK2-common.h common/pdc.h | bin
	$$(MPICC) $$(CPPFLAGS) $$(CFLAGS) $$< $$(LDFLAGS) $$(LDLIBS) -o $$@
endef
$(foreach v,1 2 3 4,$(eval $(call BUILD_Q2,$(v))))

define BUILD_Q3
bin/$(word $(1),$(Q3_NAMES)): Q3/23i-0656-C-TASK3-V$(1).c Q3/23i-0656-C-TASK3-common.h common/pdc.h | bin
	$$(MPICC) $$(CPPFLAGS) $$(CFLAGS) $$< $$(LDFLAGS) $$(LDLIBS) -o $$@
endef
$(foreach v,1 2 3,$(eval $(call BUILD_Q3,$(v))))

check: all
	./run.sh smoke

plan:
	./run.sh full --dry-run

package:
	$(PYTHON) tools/package.py

clean:
	rm -rf bin

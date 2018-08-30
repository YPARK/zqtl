
PKG := $(shell cat DESCRIPTION | awk '$$1 == "Package:" { print $$2 }')
VER := $(shell cat DESCRIPTION | awk '$$1 == "Version:" { print $$2 }')

SRC := $(wildcard src/*.cc)
HDR := $(wildcard src/*.hh)
MAN := $(wildcard man/*.Rd)

all: $(PKG)_$(VER).tar.gz

$(PKG)_$(VER).tar.gz: zqtl_R_source.R $(SRC) $(HDR) $(MAN)
	[ -f R/RcppExports.R ] || cp zqtl_R_source.R R/RcppExports.R
	R -e "options(buildtools.check = function(action) TRUE); devtools::document();"
	R CMD build .

R/RcppExports.R: zqtl_R_source.R
	cp $^ $@

check: $(PKG)_$(VER).tar.gz
	R CMD check $<

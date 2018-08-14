
PKG := $(shell cat DESCRIPTION | awk '$$1 == "Package:" { print $$2 }')
VER := $(shell cat DESCRIPTION | awk '$$1 == "Version:" { print $$2 }')

SRC := $(wildcard src/*.cc)
HDR := $(wildcard src/*.hh)
MAN := $(wildcard man/*.Rd)

all: R/RcppExports.R $(PKG)_$(VER).tar.gz

$(PKG)_$(VER).tar.gz: $(SRC) $(HDR) $(MAN) R/RcppExports.R
	R -e "roxygen2::roxygenise();"
	R CMD build .

R/RcppExports.R: zqtl_R_source.R
	cp $^ $@

check: $(PKG)_$(VER).tar.gz
	R CMD check $<

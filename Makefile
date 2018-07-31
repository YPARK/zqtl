
PKG := $(shell cat DESCRIPTION | awk '$$1 == "Package:" { print $$2 }')
VER := $(shell cat DESCRIPTION | awk '$$1 == "Version:" { print $$2 }')

SRC := $(wildcard src/*.cc)
HDR := $(wildcard src/*.hh)
RR := $(wildcard R/*.R)
MAN := $(wildcard man/*.Rd)

$(PKG)_$(VER).tar.gz: $(SRC) $(HDR) $(RR) $(MAN)
	R -e "roxygen2::roxygenise();"
	R CMD build .

check: $(PKG)_$(VER).tar.gz
	R CMD check $<

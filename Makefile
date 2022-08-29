
PKG := $(shell cat DESCRIPTION | awk '$$1 == "Package:" { print $$2 }')
VER := $(shell cat DESCRIPTION | awk '$$1 == "Version:" { print $$2 }')

SRC := $(wildcard src/*.cc)
HDR := $(wildcard src/*.hh)
MAN := $(wildcard man/*.Rd)

all: $(PKG)_$(VER).tar.gz

clean:
	rm -f src/*.o src/*.so $(PKG)_$(VER).tar.gz

$(PKG)_$(VER).tar.gz: $(SRC) $(HDR) $(MAN)
	R -e "options(buildtools.check = function(action) TRUE); roxygen2::roxygenize();"
	R CMD build .

check: $(PKG)_$(VER).tar.gz
	R CMD check $<

install: $(PKG)_$(VER).tar.gz
	R CMD INSTALL $< 

site:
	R -e "pkgdown::build_site()"

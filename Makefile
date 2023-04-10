make all:
	make -C v64

vXXX:
	make -C vXXX

v64:
	make -C v64

clean:
	make -C v64 clean


.PHONY: clean vXXX clean v64

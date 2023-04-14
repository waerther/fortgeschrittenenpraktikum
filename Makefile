make all:
	make -C v64
	make -C v61

vXXX:
	make -C vXXX

v64:
	make -C v64

v61:
	make -C v61

clean:
	make -C v64 clean
	make -C v61 clean

.PHONY: clean vXXX clean v64 clean v61

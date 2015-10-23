package = "image_utils"
version = "0.1-1"

source = {
	url = "git://github.com/adityaramesh/image_utils",
	tag = "master"
}

description = {
	summary  = "Image preprocessing utilities for Torch.",
	homepage = "https://github.com/adityaramesh/image_utils",
	license  = "BSD 3-Clause"
}

dependencies = {
	"torch >= 7.0",
	"hdf5 >= 0.0",
	"image >= 1.0"
}

build = {
	type = "command",

	-- Please tell me if you know a better way to perform the
	-- installation *without* putting all of the sources in the project
	-- root.
	build_command = [[
		rm -rf $(PREFIX)/../../../../../share/lua/5.1/image_utils;
		mkdir $(PREFIX)/../../../../../share/lua/5.1/image_utils;
		cp -R init.lua image_utils/* $(PREFIX)/../../../../../share/lua/5.1/image_utils;
	]]
}

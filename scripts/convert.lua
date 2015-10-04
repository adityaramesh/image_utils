require "lfs"
local cwd = lfs.currentdir()
package.path = package.path .. ";" .. cwd .. "/torch_utils/?.lua;" ..
	cwd ..  "/torch_utils/torch_utils/?.lua;"
package.path = package.path .. ";" .. cwd .. "/image_utils/?.lua;" ..
	cwd ..  "/image_utils/image_utils/?.lua;"

require "torch"
require "image_utils"

local cmd = torch.CmdLine()
cmd:text("Options:")
cmd:option("-input", "", "Input file path.")
cmd:option("-output", "", "Output file path.")
opt = cmd:parse(arg)

image_utils.save(opt.output, image_utils.load(opt.input))

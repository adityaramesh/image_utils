require "lfs"
require "image_utils"

local cmd = torch.CmdLine()
cmd:text("Options:")
cmd:option("-input", "", "Input file path.")
cmd:option("-output", "", "Output file path.")
opt = cmd:parse(arg)

image_utils.save(opt.output, image_utils.load(opt.input))

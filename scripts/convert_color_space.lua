require "lfs"
local cwd = lfs.currentdir()
package.path = package.path .. ";" .. cwd .. "/image_utils/?.lua;" ..
	cwd ..  "/image_utils/image_utils/?.lua"

require "image"
require "image_utils"

cmd = torch.CmdLine()
cmd:text("Options:")
cmd:option("-cs_src", "rgb", "Source color space (rgb [0, 255], nrgb [0, 1], yuv, lab).")
cmd:option("-cs_dst", "rgb", "Destination color space (rgb [0, 255], nrgb [0, 1], yuv, lab).")
cmd:option("-input", "", "Input file path.")
cmd:option("-output", "", "Output file path.")
opt = cmd:parse(arg)

local function check_cs(cs)
	local valid_cs = {rgb = true, nrgb = true, yuv = true, lab = true}

	if not valid_cs[cs] then
		error("Unrecognized color space '" .. cs .. "'.")
	end
end

check_cs(opt.cs_src)
check_cs(opt.cs_dst)

if opt.cs_src == opt.cs_dst then
	error("Source and destination color spaces are the same.")
end

if paths.filep(opt.output) then
	error("File '" .. opt.output .. "' already exists.")
end

local fn = paths.basename(opt.input)
print("Converting " .. fn .. " from " .. opt.cs_src .. " to " .. opt.cs_dst .. ".")

print("Loading data.")
local data = image_utils.load(opt.input)
local count = data.inputs:size(1)

if opt.cs_src == "rgb" then
	local t = data.inputs:type()
	if t ~= "torch.FloatTensor" and t ~= "torch.DoubleTensor" then
		data.inputs = data.inputs:double()
	end

	data.inputs:div(255)
end

for i = 1, count do
	if i % 1000 == 0 then
		print("Working on image " .. i .. " of " .. count .. ".")
	end

	if opt.cs_src == "rgb" or opt.cs_src == "nrgb" then
		if opt.cs_dst == "yuv" then
			data.inputs[i] = image.rgb2yuv(data.inputs[i])
		elseif opt.cs_dst == "lab" then
			data.inputs[i] = image.rgb2lab(data.inputs[i])
		end
	elseif opt.cs_src == "yuv" then
		if opt.cs_dst == "rgb" or opt.cs_dst == "nrgb" then
			data.inputs[i] = image.yuv2rgb(data.inputs[i])
		elseif opt.cs_dst == "lab" then
			data.inputs[i] = image.yuv2rgb(data.inputs[i])
			data.inputs[i] = image.rgb2lab(data.inputs[i])
		end
	elseif opt.cs_src == "lab" then
		if opt.cs_dst == "rgb" or opt.cs_dst == "nrgb" then
			data.inputs[i] = image.lab2rgb(data.inputs[i])
		elseif opt.cs_dst == "yuv" then
			data.inputs[i] = image.lab2rgb(data.inputs[i])
			data.inputs[i] = image.rgb2yuv(data.inputs[i])
		end
	end
end

if opt.cs_dst == "rgb" then
	data.inputs:mul(255)
end

print("Saving data to '" .. opt.output .. "'.")
image_utils.save(opt.output, data)

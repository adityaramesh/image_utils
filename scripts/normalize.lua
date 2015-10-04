require "lfs"
local cwd = lfs.currentdir()
package.path = package.path .. ";" .. cwd .. "/torch_utils/?.lua;" ..
	cwd ..  "/torch_utils/torch_utils/?.lua"
package.path = package.path .. ";" .. cwd .. "/image_utils/?.lua;" ..
	cwd ..  "/image_utils/image_utils/?.lua"

require "image"
require "image_utils"

cmd = torch.CmdLine()
cmd:text("Options:")
cmd:option("-gcn", "pixel_std", "Type of global contrast normalization (none, " ..
	"pixel_std [per-pixel standardization], image_std [per-image standardization]).")
cmd:option("-lcn", "123", "Channels on which to use local contrast normalization " ..
	"(none, 1, 2, 3, 12, 13, 23, 123).")

cmd:option("-input",  "", "Path to input file.")
cmd:option("-output", "", "Path to output file.")
cmd:option("-stats_input", "", "Path to HDF5 file with precomputed image statistics.")
cmd:option("-stats_output",  "", "Path to file in which to save image statistics.")
opt = cmd:parse(arg)

local function ensure_not_exists(fp)
	if paths.filep(fp) then
		error("File '" .. fp .. "' already exists.")
	end
end

ensure_not_exists(opt.output)
if opt.stats_output ~= "" then
	ensure_not_exists(opt.stats_output)
end

print("Loading data.")
local data = image_utils.load(opt.input)
local params = {gcn = opt.gcn, lcn = opt.lcn}

local function tensor_to_number(x)
	if x:nDimension() == 1 and x:size(1) == 1 then
		return x[1]
	end
	return x
end

local function number_to_tensor(x)
	if type(x) == "number" then
		return torch.DoubleTensor{x}
	end
	return x
end

if opt.stats_input ~= "" then
	local file = hdf5.open(opt.stats_input, 'r')
	params.mean = {}
	params.std = {}
	
	for k, v in pairs(file._rootGroup._children) do
		local mean_num = string.match(k, "mean_(%d+)")
		local std_num = string.match(k, "std_(%d+)")

		if mean_num then
			params.mean[tonumber(mean_num)] = tensor_to_number(file:read(k):all())
		elseif std_num then
			params.std[tonumber(std_num)] = tensor_to_number(file:read(k):all())
		end
	end
	file:close()
end

local mean, std = image_utils.normalize(data, params)

if opt.stats_output ~= "" then
	print("Saving statistics to '" .. opt.stats_output .. "'.")
	local file = hdf5.open(opt.stats_output, 'w')
	assert(#mean == #std)
	
	for i = 1, #mean do
		file:write("mean_" .. i, number_to_tensor(mean[i]))
		file:write("std_" .. i, number_to_tensor(std[i]))
	end
	file:close()
end

print("Saving normalized data to '" .. opt.output .. "'.")
image_utils.save(opt.output, data)

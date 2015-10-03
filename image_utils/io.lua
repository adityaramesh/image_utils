require "hdf5"
require "torch"

local function validate(inputs, targets, classes)
	assert(inputs:nDimension() == 3 or inputs:nDimension() == 4)

	if targets and classes then
		assert(inputs:size(1) == targets:size(1))
		assert(classes > 0)
		return
	elseif not (targets or classes) then
		return
	end

	error("'targets' and 'classes' must either both be present or undefined.")
end

function image_utils.load(fn)
	local ext = paths.extname(fn)

	if ext == "t7" then
		local data = torch.load(fn)
		validate(data.inputs, data.targets, data.classes)
		return data
	elseif ext == "hdf5" then
		local file = hdf5.open(fn)
		local data = {inputs = file:read("/inputs"):all()}

		if file._rootGroup._children.targets then
			data.targets = file:read("/targets"):all()
		end

		if file._rootGroup._children.classes then
			data.classes = file:read("/classes"):all()[1]
		end

		validate(data.inputs, data.targets, data.classes)
		return data
	else
		error("Unsupported file format '" .. ext .. "'.")
	end
end

function image_utils.save(fn, data)
	validate(data.inputs, data.targets, data.classes)
	local ext = paths.extname(fn)

	if ext == "t7" then
		torch.save(fn, data)
	elseif ext == "hdf5" then
		local file = hdf5.open(fn)
		file:write("/inputs", data.inputs)

		if data.targets then
			file:write("/targets", data.targets)
			file:write("/classes", torch.IntTensor{data.classes})
		end

		file:close()
	else
		error("Unsupported file format '" .. ext .. "'.")
	end
end

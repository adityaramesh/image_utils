require "torch"
require "nn"

local function validate_options(data, opt)
	-- Validate the GCN option.
	local valid_gcn = {none = true, pixel_std = true, image_std = true}
	if not valid_gcn[opt.gcn] then
		error("Invalid gcn option '" .. opt.gcn .. "'.")
	end

	-- Validate the LCN option.
	local lcn_channels = {}
	if opt.lcn ~= "none" then
		for i in string.gmatch(opt.lcn, "%d") do
			lcn_channels[tonumber(i)] = true
		end
	end

	if (opt.mean and not opt.std) or (not opt.mean and opt.std) then
		error("Either none or both of the mean and standard deviation " ..
			"must be provided.")
	end

	-- It only makes sense to do normalization when the input is of
	-- floating-point type.
	local itype = data.inputs:type()
	if not (itype == "torch.FloatTensor" or itype == "torch.DoubleTensor") then
		error("Input must be of floating-point type.")
	end

	-- Retrieve the input size information.
	local count = data.inputs:size(1)
	local channels = 0
	local width = 0
	local height = 0

	if data.inputs:nDimension() == 3 then
		channels = 1
		width = data.inputs:size(2)
		height = data.inputs:size(3)
	elseif data.inputs:nDimension() == 4 then
		assert(data.inputs:size(2) == 3)
		channels = 3
		width = data.inputs:size(3)
		height = data.inputs:size(4)
	else
		error("Input has invalid number of dimensions.")
	end

	-- Validate the channel information.
	if #lcn_channels > 1 then
		assert(channels == 3)
		assert(#lcn_channels <= channels)
	end

	for c, _ in pairs(lcn_channels) do
		assert(c >= 1 and c <= channels)
	end

	-- Validate the mean and standard deviation, if they were provided.
	if opt.gcn ~= "none" and opt.mean then
		if opt.gcn == "pixel_std" then
			local function validate(stat)
				assert(stat:type() == itype)
				assert(stat:nDimension() == 2)
				assert(stat:size(1) == width)
				assert(stat:size(2) == height)
			end

			for _, v in pairs(opt.mean) do
				validate(v)
			end

			for _, v in pairs(opt.std) do
				validate(v)
			end
		elseif opt.gcn == "image_std" then
			local function validate(stat)
				assert(type(stat) == "number")
			end

			for _, v in pairs(opt.mean) do
				validate(v)
			end

			for _, v in pairs(opt.std) do
				validate(v)
			end
		end
	end

	return lcn_channels, itype, count, channels, width, height
end

local function do_global_normalization(data, opt, itype, width, height, channels)
	-- Torch's mean and standard deviation computations are not implemented
	-- stably (same as numpy), so we convert to double first.
	local new_itype = itype
	if opt.gcn ~= "none" and itype == "torch.FloatTensor" and not opt.mean then
		new_itype = "torch.DoubleTensor"
		print("Converting input to double for GCN.")
		data.inputs = data.inputs:double()
	end

	local mean = opt.mean or {}
	local std  = opt.std or {}

	if #mean == 0 and opt.gcn == "pixel_std" then
		print("Computing pixel-wide mean and standard deviation.")

		if channels == 1 then
			mean[1] = torch.mean(data.inputs, 1)
			std[1] = torch.std(data.inputs, 1)

			assert(torch.lt(std[1], 0):sum() == 0)
			mean[1] = mean[1]:reshape(width, height)
			std[1] = std[1]:reshape(width, height)
		else
			for i = 1, channels do
				mean[i] = torch.mean(data.inputs[{{}, i}], 1)
				std[i] = torch.std(data.inputs[{{}, i}], 1)

				assert(torch.lt(std[i], 0):sum() == 0)
				mean[i] = mean[i]:reshape(width, height)
				std[i] = std[i]:reshape(width, height)
			end
		end
	elseif #mean == 0 and opt.gcn == "image_std" then
		print("Computing image-wide mean and standard deviation.")

		if channels == 1 then
			mean[1] = data.inputs:mean()
			std[1] = data.inputs:std()
			assert(std[1] > 0)
		else
			for i = 1, channels do
				mean[i] = data.inputs[{{}, i}]:mean()
				std[i] = data.inputs[{{}, i}]:std()
				assert(std[i] > 0)
			end
		end
	end

	-- If the standard deviation for a feature is below this threshold, then
	-- we will assume that it is actually zero, and will skip scaling.
	local std_tol = 1e-8

	if opt.gcn == "pixel_std" then
		print("Performing pixel-wide standardization.")

		local function fix_small_std(arr)
			local copy = arr:clone()
			local mask = torch.lt(copy, std_tol)

			local count = mask:sum()
			if count > 0 then
				print("Warning: " .. count .. " components of " ..
					"std were less than std_tol.")
			end

			copy[mask] = 1
			return copy
		end

		if channels == 1 then
			local m = mean[1]:reshape(1, width, height)
			local s = fix_small_std(std[1]:reshape(1, width, height))

			data.inputs
				:add(-1, m:expandAs(data.inputs))
				:cdiv(s:expandAs(data.inputs))
		else
			for i = 1, channels do
				print("Working on channel " .. i .. ".")	
				local m = mean[i]:reshape(1, width, height)
				local s = fix_small_std(std[i]:reshape(1, width, height))

				data.inputs[{{}, i}]
					:add(-1, m:expandAs(data.inputs[{{}, i}]))
					:cdiv(s:expandAs(data.inputs[{{}, i}]))
			end
		end
	elseif opt.gcn == "image_std" then
		print("Performing image-wide standardization.")

		if channels == 1 then
			if std[1] > std_tol then
				data.inputs:add(-mean[1]):div(std[1])
			else
				print("std[1]=" .. std[1] .. "; skipping scaling.")
			end
		else
			for i = 1, channels do
				print("Working on channel " .. i .. ".")	

				if std[i] > std_tol then
					data.inputs[{{}, i}]:add(-mean[i]):div(std[i])
				else
					print("std[" .. i .. "]=" .. std[i] ..
						"; skipping scaling.")
				end
			end
		end
	end

	-- If we converted the images from float to double earlier so that
	-- statistics for GCN could be computed more easily, then we now convert
	-- back to float.
	if itype ~= new_itype then
		print("Converting input back to float.")
		data.inputs = data.inputs:float()
	end

	return mean, std
end

local function do_local_normalization(data, opt, lcn_channels, itype, count, channels)
	print("Performing per-channel local normalization.")
	local kernel = image.gaussian1D(13)
	local module = nn.SpatialContrastiveNormalization(1, kernel, 1):type(itype)

	for i = 1, count do
		if i % 1000 == 1 then
                        print("Working on image " .. i .. " of " .. count .. ".")
                end

		for j, _ in pairs(lcn_channels) do
			data.inputs[{i, {j}}] = module:forward(data.inputs[{i, {j}}])
		end
	end
end

--
-- Applies local and global contrast normalization to the images in `data` as
-- specified by `options`. Format of `options`:
-- * `gcn` (optional): "none", "pixel_std" (per-pixel standardization),
-- "image_std" (per-image standardization) (default none).
-- * `lcn` (optional): "none", "1", "2", "3", "12", "13", "23", "123" (default
-- all channels, which corresponds to either 1 or 123, depending on whether the
-- image is in grayscale or color).
-- * `mean` (optional): mean to use for GCN. If provided, must also provide
-- `std`.
-- * `std` (optional): standard deviation to use for GCN. If provided, must also
-- provide `mean`.
--
-- Notes:
-- * `data` is modified in-place.
-- * If `gcn ~= "none"` and the mean and standard deviation are not provided,
-- then they are returned after normalization is performed. Otherwise, the
-- provided mean and standard deviation are returned.
--
function image_utils.normalize(data, opt)
	local lcn_channels, itype, count, channels, width, height =
		validate_options(data, opt)
	local mean, std = do_global_normalization(data, opt, itype, width,
		height, channels)
	do_local_normalization(data, opt, lcn_channels, itype, count, channels)

	print("Done.")
	return mean, std
end

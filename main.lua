
require 'torch'
require 'nn'
require 'image'
require 'optim'

loadcaffe_wrap = require 'loadcaffe_wrapper'
cjson = require 'cjson'
string = require 'string'

cmd = torch.CmdLine()

-- Hacking torch
cmd:option('-start_at', 1, 'index to start at – worst hack I\'ve ever written')
cmd:option('-iter', 100, 'how many images to run over – please don\'t segfault')

-- Basic options
cmd:option('-style_dir', 'data/picasso_cubism/', 'Style input directory')
cmd:option('-tmp_dir', 'tmp/', 'Directory to store vectors on disk')
cmd:option('-gpu', -1, 'Zero-indexed ID of the GPU to use; for CPU mode set -gpu = -1')

-- Other options
cmd:option('-pooling', 'max', 'max|avg')
cmd:option('-proto_file', 'models/VGG_ILSVRC_19_layers_deploy.prototxt')
cmd:option('-model_file', 'models/VGG_ILSVRC_19_layers.caffemodel')

cmd:option('-content_layers', 'relu4_2', 'layers for content')
cmd:option('-style_layers', 'relu4_1', 'layers for style') -- tbh all but relu6 and relu7, which cause size mismatches
                            -- 'relu1_1,relu2_1,relu3_1,relu4_1,relu5_1'
                            -- 'relu1_1,relu1_2,relu2_1,relu2_2,relu3_1,relu3_2,relu3_3,relu3_4,relu4_1,relu4_2,relu4_3,relu4_4,relu5_1,relu5_2,relu5_3,relu5_4'


-------------------------------------------------------------------------------------

-- Preprocess an image before passing it to a Caffe model.
-- We need to rescale from [0, 1] to [0, 255], convert from RGB to BGR,
-- and subtract the mean pixel. [jcjohnson]
function preprocess(img)
  local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
  local perm = torch.LongTensor{3, 2, 1}
  local img = img:index(1, perm):mul(256.0)
  mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  img:add(-1, mean_pixel)
  if params.gpu >= 0 then img = img:cuda() end
  return img
end

-- Returns a network that computes the CxC Gram matrix from inputs
-- of size C x H x W – jcjohnson's version
function GramMatrix()
    local net = nn.Sequential()
    net:add(nn.View(-1):setNumInputDims(2))
    local concat = nn.ConcatTable()
    concat:add(nn.Identity())
    concat:add(nn.Identity())
    net:add(concat)
    net:add(nn.MM(false, true))
    return net
end


-- utility function to reshape a tensor from M x N x ... to an MxN array
function flatten(t)
    return torch.view(t, -1) -- :storage() exposes a raw memory interface
end

-- a function to do memory optimizations by 
-- setting up double-buffering across the network.
-- this drastically reduces the memory needed to generate samples.
-- from soumith/dcgan.torch
function optimizeInferenceMemory(net)
    local finput, output, outputB
    net:apply(
        function(m)
            if torch.type(m):find('Convolution') then
                finput = finput or m.finput
                m.finput = finput
                output = output or m.output
                m.output = output
            elseif torch.type(m):find('ReLU') then
                m.inplace = true
            elseif torch.type(m):find('BatchNormalization') then
                outputB = outputB or m.output
                m.output = outputB
            end
    end)
end


function Style2Vec(cnn, gram, img, desired_layer)
    --[[ runs img through cnn, saving the output tensor at each of style_layers

    -- FOR NOW, only returns relu4_1

    relu1_1 : FloatTensor - size: 64x64
    relu1_2 : FloatTensor - size: 64x64
    relu2_1 : FloatTensor - size: 128x128
    relu2_2 : FloatTensor - size: 128x128
    relu3_1 : FloatTensor - size: 256x256
    relu3_2 : FloatTensor - size: 256x256
    relu3_3 : FloatTensor - size: 256x256
    relu3_4 : FloatTensor - size: 256x256
    relu4_1 : FloatTensor - size: 512x512
    relu4_2 : FloatTensor - size: 512x512
    relu4_3 : FloatTensor - size: 512x512
    relu4_4 : FloatTensor - size: 512x512
    relu5_1 : FloatTensor - size: 512x512
    relu5_2 : FloatTensor - size: 512x512
    relu5_3 : FloatTensor - size: 512x512
    relu5_4 : FloatTensor - size: 512x512
    
    Returns a Lua table with the above key-value pairs. 

    
    --]]
    
    local next_style_idx = 1
    local net = nn.Sequential()
    local style_vec = {}
    local style_layers = params.style_layers:split(',')

    -- Negatory, need to actually build up the net in order to get forward output
    -- from a specific layer (afaik)

    -- for i = 1, #cnn do
    --     local layer = cnn:get(i)
    --     local layer_name = layer.name
    --     if (layer_name == desired_layer) then
    --         local gram = GramMatrix():float()
    --         if params.gpu >= 0 then gram = gram:cuda() end 
    --         cnn:forward(img)
    --         cnn:get(i).output



    -- Build up net from cnn
    
    for i = 1, #cnn do

        if next_style_idx <= #style_layers then
            local layer = cnn:get(i)
            local layer_name = layer.name

            if params.gpu >= 0 then layer = layer:cuda() end

            net:add(layer)
            
            -- now to grab style layers
            
            if (layer_name == desired_layer) then
                local target_features = net:forward(img)

                local target_i = gram:forward(target_features)
                target_i:div(target_features:nElement())
                
                -- hack to do only one layer instead of all of them

                gram = nil
                net = nil
                collectgarbage(); collectgarbage()
                return flatten(target_i):totable() --:totable() might be causing problems

                -- original code below

                -- style_vec[layer_name] = torch.totable(flatten(target_i))
                -- next_style_idx = next_style_idx + 1

                -- end hack
     
            end
        end
    end

    error("Couldn't find layer " .. desired_layer)
    collectgarbage(); collectgarbage()
    return false
end


function load_json(filename, file)
    local str = torch.load(params.tmp_dir .. filename .. '.json', 'ascii')
    return cjson.decode(str)
end


function save_json(filename, file)        
    local json_string = cjson.encode(file)
    torch.save(params.tmp_dir .. filename .. '.json', json_string, 'ascii')

    return true
end


function cached(label) -- is it cached? t/f
    for f in paths.iterfiles(params.tmp_dir) do
        if f == filename then
            -- print(filename .. 'already exists')
            return true
        end
    end
    return false
end


-----------------------------------------------------------------------------------


params = cmd:parse(arg)
if paths.dir(params.tmp_dir) == nil then paths.mkdir(params.tmp_dir) end


-- gpu

if params.gpu >= 0 then
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(params.gpu + 1)
else
    params.backend = 'nn-cpu'
end


-- load style_images

style_images = {}
sorted = {}

for f in paths.iterfiles(params.style_dir) do    
    if string.match(f, '.jpg') then

        -- print('processing ' .. f)

        local img = image.load(params.style_dir .. f)
        img = preprocess(img):float()

        if params.gpu >= 0 then img = img:cuda() end
        label = string.split(f, '.jpg')[1]

        table.insert(sorted, label)
        style_images[label] = img

    end
end

table.sort(sorted)
for i,n in ipairs(sorted) do print(i, n) end

collectgarbage(); collectgarbage()
print(collectgarbage('count'))


-- load caffe network image

local cnn = loadcaffe_wrap.load(params.proto_file, params.model_file, params.backend):float()
local gram = GramMatrix():float()
if params.gpu >= 0 then 
    cnn = cnn:cuda()
    gram = gram:cuda() 
end
optimizeInferenceMemory(cnn)



collectgarbage(); collectgarbage()
print(collectgarbage('count'))

-- Run Style2Vec on image by image

-- local vecs = {}
local ct = 1

i = params.start_at


while (i < #sorted) do
    label = sorted[i]

    io.write(ct .. ' ' .. label .. ':\t')        --      .. params.style_layers .. ' ...' 
    
    local image = style_images[label]
    local vec = Style2Vec(cnn, gram, image, 'relu4_1')
    vec = cjson.encode(vec)

    -- vecs[i] = vec['relu4_1']

    -- Let's try saving each vector individually, as opposed to L243 (+13)
    torch.save(params.tmp_dir .. label .. '.json', vec, 'ascii')
    
    io.write(' Done!\n')
    
    ct = ct + 1
    if ct > params.iter then break end
    collectgarbage(); collectgarbage()
    print(collectgarbage('count'))

    i = i + 1
end

-- for i, label in ipairs(sorted) do
--     io.write(label .. ':\t')        --      .. params.style_layers .. ' ...' 
    
--     local image = style_images[label]
--     local vec = Style2Vec(cnn, gram, image, 'relu4_1')
--     -- local table = vec
--     -- vec = cjson.encode(table)

--     -- table = nil
    
--     -- vecs[i] = vec['relu4_1']

--     -- Let's try saving each vector individually, as opposed to L243 (+13)
--     torch.save(params.tmp_dir .. label .. '.json', vec, 'ascii')
    
--     io.write(' Done!\n')
    
--     ct = ct + 1
--     if ct > 5 then break end
--     collectgarbage(); collectgarbage()
--     print(collectgarbage('count'))
-- end

-- store our output
torch.save(params.tmp_dir .. 'sorted.json', cjson.encode(sorted), 'ascii')
-- torch.save(params.tmp_dir .. 'vecs.json', cjson.encode(vecs), 'ascii')

-- for i, n in ipairs(sorted) do
--     if vecs[n] ~= nil then
--         print(n)
--     end
-- end


-- clean up clean up
-- vecs = nil
cnn = nil
style_images = nil
collectgarbage()


--------------------------------------------------------------------------------


-- down here be monsters

-- local ct = 1
-- local size = 262144 -- 512^2 - relu4_1

-- local tensors = nil
-- local labels = torch.String

-- for f in paths.iterfiles(params.tmp_dir) do
--     local table = torch.load(params.tmp_dir .. f, params.cache_format)
--     t = flatten(table['relu4_1']:double())
    
--     if tensors then
--         tensors = torch.cat(tensors, t)
--     else
--         tensors = t
--     end
        
--     ct = ct + 1    
-- end

-- views = torch.view(tensors, -1, size)

-- print(#views)


-- local m = require 'manifold'
-- p = m.embedding.tsne(vecs, {dim=2, perplexity=8}) -- THIS DOESN'T WORK because vecs is a table now

-- function CosineSimilarity(x, y)
--     local net = nn.Sequential()
--     net:add(nn.CosineDistance())
--     return net:forward({x, y})
-- end

-- function StyleDistance(x, y, sorted_layers)
--     -- this function will return the distance from each layer, assuming x and y
--     -- x["relu2_1 "] = torch.FloatTensor
    
--     for _, i in ipairs(sorted_layers) do -- can you tell I'm new to Lua?
--         local distance_vector = CosineSimilarity(x[i]:double(), y[i]:double())
--         local avg_distance = torch.mean(distance_vector)
        
--         local msg ='Distance at layer %s is: %f'
--         print(string.format(msg, i, avg_distance))
--     end
    
-- end


-- -- -- this is a little embarassing, no?
-- -- local labels = params.style_layers:split(',')
-- -- table.sort(labels)

-- StyleDistance(style_vecs['haring_bw.jpg'], style_vecs['haring_bw.jpg'], labels)
-- -- x = torch.Tensor({1, 2, 3})
-- -- y = torch.Tensor({4, 5, 6})
-- -- print(CosineSimilarity(x, y))

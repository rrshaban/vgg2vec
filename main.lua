
require 'torch'
require 'nn'
require 'image'
require 'optim'

loadcaffe_wrap = require 'loadcaffe_wrapper'
cjson = require 'cjson'
string = require 'string'

cmd = torch.CmdLine()

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

-- Preprocess an image before passing it to a Caffe model.
-- We need to rescale from [0, 1] to [0, 255], convert from RGB to BGR,
-- and subtract the mean pixel. [jcjohnson]
function preprocess(img)
  local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
  local perm = torch.LongTensor{3, 2, 1}
  img = img:index(1, perm):mul(256.0)
  mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  img:add(-1, mean_pixel)
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

function Style2Vec(img, cnn)
    --[[ runs img through cnn, saving the output tensor at each of style_layers

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
    
    -- Build up net from cnn
    
    for i = 1, #cnn do
        if next_style_idx <= #style_layers then
            local layer = cnn:get(i)
            local layer_name = layer.name
            local layer_type = torch.type(layer)
            local is_pooling = (layer_type == 'cudnn.SpatialMaxPooling' or layer_type == 'nn.SpatialMaxPooling')
            
            -- add layers to net from cnn, replacing max-pooling if necessary [jcjohnson]
            if is_pooling and params.pooling == 'avg' then
                local msg = 'Replacing max pooling at layer %d with average pooling'
                print(string.format(msg, i))
                assert(layer.padW == 0 and layer.padH == 0)
                -- kWxkH regions by step size dWxdH
                local kW, kH = layer.kW, layer.kH
                local dW, dH = layer.dW, layer.dH
                local avg_pool_layer = nn.SpatialAveragePooling(kW, kH, dW, dH):float()
                if params.gpu >= 0 then avg_pool_layer:cuda() end
                net:add(avg_pool_layer)
            else
                net:add(layer)
            end
            
            -- now to grab style layers
            
            if (layer_name == style_layers[next_style_idx]) then
                    
                local gram = GramMatrix():float()
                if params.gpu >= 0 then gram = gram:cuda() end
                local target_features = net:forward(img)
                local target_i = gram:forward(target_features)
                
                target_i:div(target_features:nElement())
                
                style_vec[layer_name] = torch.totable(flatten(target_i))
                -- itorch.image(target_i) -- YA THIS IS THE VECTOR!!!
                                
                next_style_idx = next_style_idx + 1
            end
        end
    end
    
    collectgarbage()
    return style_vec
end

function load_json(filename, file)
    local str = torch.load(params.tmp_dir .. filename .. '.json', 'ascii')
    return cjson.decode(str)
end

function cache_json(filename, file)    
    if paths.dir(tmp) == nil then paths.mkdir(tmp) end
    
    json_string = cjson.encode(file)
    torch.save(params.tmp_dir .. filename .. '.json', json_string, 'ascii')
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

-- let's get started

-- local arg = {} -- when running from cli, this will be defined
params = cmd:parse(arg)
print(params)

-- load caffe network image
cnn = loadcaffe_wrap.load(params.proto_file, params.model_file, params.backend):float()

if params.gpu >= 0 then cnn:cuda() end

collectgarbage()

-- load style_images

style_images = {}
sorted = {}

for f in paths.iterfiles(params.style_dir) do    
    if string.match(f, '.jpg') then

        local img = image.load(params.style_dir .. f)
        img = preprocess(img):float()

        if params.gpu >= 0 then img = img:cuda() end
        label = string.split(f, '.jpg')[1]

        table.insert(sorted, label)
        style_images[label] = img
    end
end

-- print(style_images)
table.sort(sorted)
for i,n in ipairs(sorted) do print(n) end

collectgarbage()
print(collectgarbage('count'))


-- Run Style2Vec

vecs = {}
local ct = 1

for i, label in ipairs(sorted) do
    collectgarbage()
    io.write(label .. ':\t' .. params.style_layers .. ' ...' ) 
    
    local image = style_images[label]
    local vec = Style2Vec(image, cnn)
    
    vecs[i] = vec['relu4_1']
    
    io.write(' Done!\n')
    
    ct = ct + 1
    -- if ct > 2 then break end
end


-- store our output
torch.save(params.tmp_dir .. 'sorted.json', cjson.encode(sorted), 'ascii')
torch.save(params.tmp_dir .. 'vecs.json', cjson.encode(vecs), 'ascii')

for i, n in ipairs(sorted) do
    if vecs[n] ~= nil then
        print(n)
    end
end


-- clean up clean up
vecs = nil
cnn = nil
style_images = nil
collectgarbage()


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

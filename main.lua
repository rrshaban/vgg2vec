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
cmd:option('-img_size', 512, 'all images will be resized to this max dimension' )

-- Basic options
cmd:option('-style_dir', 'data/picasso_cubism/', 'Style input directory')
cmd:option('-tmp_dir', 'tmp/', 'Directory to store vectors on disk')
cmd:option('-gpu', -1, 'Zero-indexed ID of the GPU to use; for CPU mode set -gpu = -1')

-- Other options
cmd:option('-pooling', 'max', 'max|avg')
cmd:option('-proto_file', 'models/VGG_ILSVRC_19_layers_deploy.prototxt')
cmd:option('-model_file', 'models/VGG_ILSVRC_19_layers.caffemodel')

cmd:option('-content_layers', 'relu4_2', 'layers for content')
cmd:option('-style_layers', 'relu1_1,relu2_1,relu3_1,relu4_1,relu5_1', 'layers for style') -- tbh all but relu6 and relu7, which cause size mismatches
                            -- 'relu1_1,relu2_1,relu3_1,relu4_1,relu5_1'
                            -- 'relu1_1,relu1_2,relu2_1,relu2_2,relu3_1,relu3_2,relu3_3,relu3_4,relu4_1,relu4_2,relu4_3,relu4_4,relu5_1,relu5_2,relu5_3,relu5_4'


-------------------------------------------------------------------------------------


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
    return torch.view(t, -1)
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


function Style2Vec(cnn, gram, img)
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
    local style_layers = params.style_layers:split(',')
    
    local style_vector = nil 

    -- THIS GUY THIS GUY THIS GUY THIS GUY
    -- nn.JoinTable(1):forward{x, y, x}:float()

    -- Build up net from cnn
    
    for i = 1, #cnn do

        if next_style_idx <= #style_layers then
            local layer = cnn:get(i)
            local layer_name = layer.name

            if params.gpu >= 0 then layer = layer:cuda() end

            net:add(layer)
            
            -- now to grab style layers
            
            if (layer_name == style_layers[next_style_idx]) then
                local target_features = net:forward(img)
                local target_i = gram:forward(target_features)
                target_i:div(target_features:nElement())
                
                -- add the current gram matrix (flattened) to style_vector
                local curr = flatten(target_i):float()
                if style_vector == nil then
                    style_vector = curr
                else
                    style_vector = nn.JoinTable(1):forward({style_vector, curr}):float()
                end

                next_style_idx = next_style_idx + 1     
            end
        end
    end

    collectgarbage(); collectgarbage()
    return style_vector
end


-- function load_json(filename, file)
--     local str = torch.load(params.tmp_dir .. filename .. '.json', 'ascii')
--     return cjson.decode(str)
-- end


function save_json(filename, file)        
    local json_string = cjson.encode(file)
    torch.save(params.tmp_dir .. filename .. '.json', json_string, 'ascii')

    return true
end


-- function cached(label) -- is it cached? t/f
--     for f in paths.iterfiles(params.tmp_dir) do
--         if f == filename then
--             -- print(filename .. 'already exists')
--             return true
--         end
--     end
--     return false
-- end

function load(label) -- load and preprocess image

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

    local img = image.load(params.style_dir .. label .. '.jpg')
    img = image.scale(img, params.img_size, 'bilinear')
    img = preprocess(img):float()
    return img
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


-- get sorted
sorted = {}

for f in paths.iterfiles(params.style_dir) do    
    if string.match(f, '.jpg') then
        label = string.split(f, '.jpg')[1]
        table.insert(sorted, label)
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

local ct = 1

i = params.start_at

local vecs = nil

while (i < #sorted) do
    label = sorted[i]

    io.write(ct .. ' ' .. label .. ':\t')        --      .. params.style_layers .. ' ...' 
    
    local image = load(label)
    local vec = Style2Vec(cnn, gram, image)

    if vecs == nil then
        vecs = vec 
    else
        vecs = nn.JoinTable(1):forward({vecs, vec}):float()
    end

    io.write(' Done!\n')
    
    i = i + 1
    ct = ct + 1
    if ct > params.iter then break end
    collectgarbage(); collectgarbage()
    print(collectgarbage('count'))
end


-- clean up a little
cnn = nil
style_images = nil
collectgarbage(); collectgarbage()

-- pass our vecs through t-SNE
vecs = vecs:view(ct - 1, -1)
print(#vecs)

local m = require 'manifold'
local p = m.embedding.tsne(vecs:double(), {dim=2, perplexity=8})

print(#p)
print(p)

assert(save_json('sorted', sorted))
assert(save_json('embedding', p:totable()))

--------------------------------------------------------------------------------
-- down here be monsters

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

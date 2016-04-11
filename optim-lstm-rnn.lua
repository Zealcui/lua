require 'rnn'
require 'optim'
require 'cunn'
dofile('loadUCR.lua')
metrics = require 'metrics'

cmd = torch.CmdLine()
cmd:text()
cmd:text("LSTM using Optim RNN for UCR Time Series")
cmd:text()
cmd:text("Options")
cmd:option('-max_iter',200, "Max epoch for training")
cmd:text()

opt = cmd:parse(arg)

function predResult(val)
    res = val:clone()
    res[val:le(0)] = -1
    res[val:ge(0)] = 1
    return res
end

-- hyper-parameters 
batch_size = 8
rho = 4 -- sequence length : n - input_size + 1
hidden_size = 7
nIndex = 10
input_size = 6
output_size = 1
lr = 0.1

-- build simple recurrent neural network
function createModel(input_size, hidden_size, output_size)
    local rnn = nn.Sequential()
            :add(nn.Linear(input_size, hidden_size))
            :add(nn.LSTM(hidden_size, hidden_size))
            :add(nn.LSTM(hidden_size, hidden_size))
            :add(nn.Linear(hidden_size, output_size))
            :add(nn.Tanh())

    rnn = nn.Sequencer(rnn)
    -- build criterion
    criterion = nn.MSECriterion()
    seqC = nn.SequencerCriterion(criterion)

    return rnn, seqC
end

-- build dummy dataset (task is to predict next item, given previous)
function loadData()
    local xr, yr= loadUCR('ItalyPowerDemand_TRAIN')
    local xe, ye= loadUCR('ItalyPowerDemand_TEST')
    yr[yr:le(0.1)] = -1
    ye[ye:le(0.1)] = -1

    local data = {}
    data['xr'] = xr
    data['yr'] = yr
    data['xe'] = xe
    data['ye'] = ye

    -- shuffle training set
    local shuffle_idx = torch.randperm(data.xr.size(1), 'torch.LongTensor')
    data.xr = data.xr:index(1, shuffle_idx)
    data.yr = data.yr:index(1, shuffle_idx)

    -- normalization
    
    -- validation set
    local n_valid = math.floor(data.xr:size(1) * 0.2)
    local n_train = data.xr:size(1) = n_valid
    data['xv'] = data.xr:sub(n_train + 1, data.xr:size(1))
    data['yv'] = data.yr:sub(n_train + 1, data.yr:size(1))
    data['xr'] = data.xr:sub(1, n_train)
    data['yr'] = data.yr:sub(1, n_train)
    
    return data
end

-- x:cuda()
-- y:cuda()

-- training

function train(model, criterion, W, grad, data)
    model:training()
    
end

print('training...')
 
local iteration = 1
while iteration < opt.max_iter do     
    -- 1. create a sequence of rho time-steps
    local inputs, targets = {}, {}
    for step=1,rho do
      -- a batch of inputs
      inputs[step] = x[{{}, {input_size * (step -1) + 1 , input_size * step}}]:cuda()
      -- incement indices
      targets[step] = y:cuda()
    end
     
   -- 2. forward sequence through rnn
   
   rnn:forget() -- forget all past time-steps
   
   local outputs, err = {}, 0
   -- for step=1,rho do
   --    outputs[step] = rnn:forward(inputs[step])
   --    err = err + criterion:forward(outputs[step], targets[step])
   -- end
   local out = rnn:forward(inputs)
   err = err + seqC:forward(out, targets)
   gradOut = seqC:backward(out, targets)
   rnn:backward(inputs, gradOut)
   rnn:updateParameters(0.05)
   rnn:zeroGradParameters() 
 
   print(string.format("Iteration %d ; NLL err = %f ", iteration, err/67.0))
   
   iteration = iteration + 1
end

rnn:forget()
x, y= loadUCR('ItalyPowerDemand_TEST')
y[y:le(0.1)] = -1

local inputs, targets = {}, {}
for step=1,rho do
  -- a batch of inputs
  inputs[step] = x[{{}, {input_size * (step -1) + 1 , input_size * step}}]:cuda()
  -- incement indices
  targets[step] = y:cuda()
end
 
-- 2. forward sequence through rnn

rnn:zeroGradParameters() 
rnn:forget() -- forget all past time-steps

local outputs, err = {}, 0
outputs = rnn:forward(inputs)

print('label:')
pred = (outputs[4])
pred:resize(pred:size(1))

roc_points, thresholds = metrics.roc.points(pred:double(), y:double())
area = metrics.roc.area(roc_points)
print('ROC: ' .. area)
pred = predResult(pred)
res = 0
for i = 1, y:size(1) do
    res = res + math.abs((y[i] - pred[i])/2)
end
print('Err: ' .. res/y:size(1))
-- print((y - pred):mean())
--
--
local function evaluation(type_eval, model, confusion)
    if type_eval ~= 'r' and type_eval ~= 'e' and type_eval ~= 'v' then
        error('Unrecognized Evaluation Instruction')
    end

    model:evaluate()

    local N = data['x', .. type_eval]:size(1)
    local err = 0

    local inputs = data
local function reportErr(data, model, confusion)
    local best_valid = math.huge
    local best_test = math.huge
    local best_train = math.huge
    local best_epoch = math.huge

    local function report(t)
        local err_e = evalution('e', data, model, confusion)
        local err_v = evalution('v', data, model, confusion)
        local err_r = evalution('r', daat, model, confusion)
        print('-------------Eopch: ' .. t .. ' of ' .. opt.max_opoch)
        print(string.format('Current Errors: test: %.4f | valid: %.4f | train: %.4f',
                            err_e, err_v, err_r))
        if best_valid > err_v then
            best_valid = err_v
            best_train = err_r
            best_test = err_e
            best_epoch = t
        end
        print(string.format('Optima achieved at epoch %d: test: %.4f, valid: %.4f',
                            best_epoch, best_test, best_valid))
    end
end
-------------------------------------------------------------------------------
--                           Main Function                                   --
-------------------------------------------------------------------------------

local function main()
    local opt = commandLine()
    torch.setdefaulttensortype('torch.FloatTensor')
    local data = loadData()
    
    local model, criterion = createModel()

    local confusion = optim.ConfusionMatrix(1)
    local W, grad = model:getParameters()

    print('the number of parameters is ' .. W:nElement())

    local report = reportErr(data, model, confusion)
end

require 'rnn'
require 'cunn'
dofile('loadUCR.lua')
metrics = require 'metrics'

cmd = torch.CmdLine()
cmd:text()
cmd:text("Simple LSTM RNN for UCR Time Series")
cmd:text()
cmd:text("Options")
cmd:option('-max_epoch',200, "Max epoch for training")
cmd:option('-gpu', false, "USE GPU")
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
rnn = nn.Sequential()
    :add(nn.Linear(input_size, hidden_size))
    :add(nn.LSTM(hidden_size, hidden_size))
    :add(nn.LSTM(hidden_size, hidden_size))
    :add(nn.Linear(hidden_size, output_size))
    :add(nn.Tanh())

rnn = nn.Sequencer(rnn)
lr_model = nn.Sequential()
lr_model:add(nn.Linear(rho, output_size))
lr_model:add(nn.Tanh())

model = nn.Sequential()
model:add(rnn)
model:add(nn.JoinTable(2))
model:add(lr_model)
print(model)


--print(rnn)

-- build criterion

--criterion = nn.BCECriterion():cuda()
criterion = nn.MSECriterion()
--seqC = nn.SequencerCriterion(criterion)
--seqC:cuda()

if opt.gpu then
    rnn:cuda()
    lr_model:cuda()
    model:cuda()
    criterion:cuda()
end
-- build dummy dataset (task is to predict next item, given previous)
x, y= loadUCR('ItalyPowerDemand_TRAIN')
y[y:le(0.1)] = -1
y[y:ge(0.1)] = 1

-- x:cuda()
-- y:cuda()

-- training

print('training...')
 
local iteration = 1
model:training()

while iteration < opt.max_epoch do     
    -- 1. create a sequence of rho time-steps
    local inputs, targets = {}, {}
    for step=1,rho do
      if opt.gpu then
          inputs[step] = x[{{}, {input_size * (step -1) + 1 , input_size * step}}]:cuda()
          -- incement indices
          targets[step] = y:cuda()
      else
          -- a batch of inputs
          inputs[step] = x[{{}, {input_size * (step -1) + 1 , input_size * step}}]
          -- incement indices
          targets[step] = y
      end
    end
     
   -- 2. forward sequence through rnn
   
   rnn:forget() -- forget all past time-steps
   
   local outputs, err = {}, 0
   -- for step=1,rho do
   --    outputs[step] = rnn:forward(inputs[step])
   --    err = err + criterion:forward(outputs[step], targets[step])
   -- end
   --[[
   local out = rnn:forward(inputs)
   local out_tensor = nn.JoinTable(2):forward(out)
   if opt.gpu then
     out_tensor:cuda()
   end
   local prediction = lr_model:forward(out_tensor)
   --print(prediction)
   err = err + criterion:forward(prediction, targets[rho])
   gradOut = criterion:backward(prediction, targets[rho])
   gradLR = lr_model:backward(out_tensor, gradOut)
   lr_model:updateParameters(0.1)
   lr_model:zeroGradParameters()
   gradLR_table = {}
   for i = 1, gradLR:size(2) do
       table.insert(gradLR_table, gradLR[{{},i}])
   end
   rnn:backward(inputs, gradLR_table)
   rnn:updateParameters(0.05)
   rnn:zeroGradParameters() 
   --]]
   local prediction = model:forward(inputs)
   err = criterion:forward(prediction, targets[rho])
   local gradOut = criterion:backward(prediction, targets[rho])
   local gradInput = model:backward(inputs, gradOut)
   model:updateParameters(0.05)
   model:zeroGradParameters()

 
   print(string.format("Iteration %d ; NLL err = %f ", iteration, err/1.))
   
   iteration = iteration + 1
end

rnn:forget()
x, y= loadUCR('ItalyPowerDemand_TEST')
y[y:le(0.1)] = -1

local inputs, targets = {}, {}
for step=1,rho do
  -- a batch of inputs
  if opt.gpu then
      inputs[step] = x[{{}, {input_size * (step -1) + 1 , input_size * step}}]:cuda()
      -- incement indices
      targets[step] = y:cuda()
  else
      inputs[step] = x[{{}, {input_size * (step -1) + 1 , input_size * step}}]
      -- incement indices
      targets[step] = y
  end
end
 
-- 2. forward sequence through rnn

--rnn:zeroGradParameters() 
rnn:forget() -- forget all past time-steps

local outputs, err = {}, 0
--[[
local out = rnn:forward(inputs)
local out_tensor = nn.JoinTable(2):forward(out)
if opt.gpu then
 out_tensor:cuda()
end
local pred = lr_model:forward(out_tensor)
--]]
local pred = model:forward(inputs)
pred:resize(pred:size(1))
 
print('label:')
print(pred)

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

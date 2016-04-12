require 'rnn'
require 'cunn'
dofile('loadUCR.lua')
metrics = require 'metrics'

cmd = torch.CmdLine()
cmd:text()
cmd:text("Simple LSTM RNN for UCR Time Series")
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
rnn = nn.Sequential()
    :add(nn.Linear(input_size, hidden_size))
    :add(nn.LSTM(hidden_size, hidden_size))
    :add(nn.LSTM(hidden_size, hidden_size))
    :add(nn.Linear(hidden_size, output_size))
    :add(nn.Tanh())

rnn = nn.Sequencer(rnn)
rnn:cuda()

print(rnn)

-- build criterion

--criterion = nn.BCECriterion():cuda()
criterion = nn.MSECriterion()
seqC = nn.SequencerCriterion(criterion)
seqC:cuda()

-- build dummy dataset (task is to predict next item, given previous)
x, y= loadUCR('ItalyPowerDemand_TRAIN')
y[y:le(0.1)] = -1

-- x:cuda()
-- y:cuda()

-- training

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

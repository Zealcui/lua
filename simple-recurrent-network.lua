require 'rnn'
require 'cunn'
dofile('loadUCR.lua')

function predResult(val)
    res = val:clone()
    res[val:le(0.5)] = 0
    res[val:ge(0.5)] = 1
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
local r = nn.Recurrent(
    hidden_size, nn.Linear(input_size, hidden_size), 
   nn.Linear(hidden_size, hidden_size), nn.Sigmoid(), 
   rho
)

local rnn = nn.Sequential()
   :add(r)
   :add(nn.Linear(hidden_size, output_size))
   :add(nn.Sigmoid())

-- wrap the non-recurrent module (Sequential) in Recursor.
-- This makes it a recurrent module
-- i.e. Recursor is an AbstractRecurrent instance
rnn = nn.Recursor(rnn, rho)
rnn:cuda()
print(rnn)

-- build criterion

--criterion = nn.BCECriterion():cuda()
criterion = nn.MSECriterion():cuda()

-- build dummy dataset (task is to predict next item, given previous)
x, y= loadUCR('ItalyPowerDemand_TRAIN')

-- x:cuda()
-- y:cuda()

-- training

print('training...')
 
local iteration = 1
while iteration < 30000 do     
    -- 1. create a sequence of rho time-steps
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
   for step=1,rho do
      outputs[step] = rnn:forward(inputs[step])
      err = err + criterion:forward(outputs[step], targets[step])
   end
   
   print(string.format("Iteration %d ; NLL err = %f ", iteration, err))

   -- 3. backward sequence through rnn (i.e. backprop through time)
   
   local gradOutputs, gradInputs = {}, {}
   for step=rho,1,-1 do -- reverse order of forward calls
      gradOutputs[step] = criterion:backward(outputs[step], targets[step])
      gradInputs[step] = rnn:backward(inputs[step], gradOutputs[step])
   end
   
   -- 4. update
   
   rnn:updateParameters(lr)
   
   iteration = iteration + 1
end
rnn:forget()
x, y= loadUCR('ItalyPowerDemand_TEST')

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
for step=1,rho do
  outputs[step] = rnn:forward(inputs[step])
  err = err + criterion:forward(outputs[step], targets[step])
end

print('label:')
pred = predResult(outputs[4])
pred:resize(pred:size(1))

print(pred:size())
print(y:size())
res = 0
for i = 1, y:size(1) do
    res = res + math.abs((y[i] - pred[i]))
end
print(res/y:size(1))
-- print((y - pred):mean())

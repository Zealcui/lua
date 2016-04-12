require 'rnn'
require 'optim'
require 'cunn'
dofile('loadUCR.lua')
metrics = require 'metrics'

local function commandLine()
    cmd = torch.CmdLine()
    cmd:text()
    cmd:text('LSTM using Optim RNN for UCR Time Series')
    cmd:text()
    cmd:text('Options')
    cmd:option('-seed', 1234, 'fixed input seed for repeatable experiments')
    cmd:option('-max_epoch',200, 'Max epoch for training')
    cmd:option('-batch_size', 8, 'Mini Batch Size')
    cmd:option('-rho', 4, 'RNN steps')
    cmd:option('-hidden_size', 7, 'hidden layer size')
    cmd:option('-input_size', 6, 'input size')
    cmd:option('-output_size', 1, 'output size')
    cmd:option('-learning_rate', 0.1, 'learning rate at t=0') 
    cmd:option('-momentum', 0.3, 'momentum (SGD only)')
    cmd:option('-decay_lr', 1e-4, 'learning rate decay')
    cmd:text()

    local opt = cmd:parse(arg)
    torch.manualSeed(opt.seed)
    return opt
end

function predResult(val)
    res = val:clone()
    res[val:le(0)] = -1
    res[val:ge(0)] = 1
    return res
end

function predResult(val)
    res = val:clone()
    res[val:le(0)] = -1
    res[val:ge(0)] = 1
    return res
end

-- build simple recurrent neural network
function createModel(opt)
    local input_size, hidden_size, output_size = opt.input_size, opt.hidden_size, opt.output_size
    local rnn = nn.Sequential()
            :add(nn.Linear(input_size, hidden_size))
            :add(nn.LSTM(hidden_size, hidden_size))
            :add(nn.LSTM(hidden_size, hidden_size))
            :add(nn.Linear(hidden_size, output_size))
            :add(nn.Tanh())

    rnn = nn.Sequencer(rnn)
    -- build criterion
    local criterion = nn.MSECriterion()
    local seqC = nn.SequencerCriterion(criterion)

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
    local shuffle_idx = torch.randperm(data.xr:size(1), 'torch.LongTensor')
    data.xr = data.xr:index(1, shuffle_idx)
    data.yr = data.yr:index(1, shuffle_idx)

    -- normalization
    
    -- validation set
    local n_valid = math.floor(data.xr:size(1) * 0.2)
    local n_train = data.xr:size(1) - n_valid
    data['xv'] = data.xr:sub(n_train + 1, data.xr:size(1))
    data['yv'] = data.yr:sub(n_train + 1, data.yr:size(1))
    data['xr'] = data.xr:sub(1, n_train)
    data['yr'] = data.yr:sub(1, n_train)
    
    return data
end

-- x:cuda()
-- y:cuda()

-- training

function train(model, criterion, W, grad, data, opt)
    model:training()
    model:forget()

    local n_train = data.xr:size(1)

    local inputs, targets = {},{}

    for step = 1, opt.rho do
        inputs[step] = data.xr[{{}, {opt.input_size * (step -1) + 1, opt.input_size * step}}]:cuda()
        targets[step] = data.yr:cuda()
    end
    
    --implement minibatch later

    function feval(x)
        assert(x == W)
        grad:zero()
     
        local prediction = model:forward(inputs)
        local loss_w     = criterion:forward(prediction, targets)
        local gradOut    = criterion:backward(prediction, targets)
        model:backward(inputs, gradOut)

        loss_w = loss_w / n_train -- adjust for train size

        return loss_w, grad
    end
    opt.optimizer(feval, W, opt.optim_config)
    
    -- local prediction = model:forward(inputs)
    -- local loss_w     = criterion:forward(prediction, targets)
    -- local gradOut    = criterion:backward(prediction, targets)
    -- model:backward(inputs, gradOut)

    -- model:updateParameters(0.05)
    -- model:zeroGradParameters()
    
end

local function evaluation(type_eval, data, model, opt, confusion)
    if type_eval ~= 'r' and type_eval ~= 'e' and type_eval ~= 'v' then
        error('Unrecognized Evaluation Instruction')
    end

    --model:evaluate()
    model:forget()

    local name_x = 'x' .. type_eval
    local name_y = 'y' .. type_eval
    local N = data[name_x]:size(1)
    local err = 0

    local inputs, targets = {}, data[name_y]

    for step = 1, opt.rho do
        inputs[step] = data[name_x][{{}, {opt.input_size * (step -1) + 1, opt.input_size * step}}]:cuda()
    end

    local outputs = model:forward(inputs)
    local pred = outputs[opt.rho]
    local n_pred = pred:size(1)
    pred = predResult(pred:resize(n_pred))

    -- confusion:batchAdd(pred, targets)

    -- confusion:updateValids()
    -- print(confusion)

    -- err = 1 - confusion.totalValid
    -- confusion:zero()
    for i = 1, n_pred do
        err = err + math.abs(pred[i] - targets[i])/2
    end
    return err/n_pred
end

local function reportErr(data, model, opt, confusion)
    local best_valid = math.huge
    local best_test = math.huge
    local best_train = math.huge
    local best_epoch = math.huge

    local function report(t)
        local err_r = evaluation('r', data, model, opt, confusion)
        local err_v = evaluation('v', data, model, opt, confusion)
        local err_e = evaluation('e', data, model, opt, confusion)
        print('-------------Eopch: ' .. t .. ' of ' .. opt.max_epoch)
        print(string.format('Current Errors: test: %.4f | valid: %.4f | train: %.4f',
                            err_e, err_v, err_r))
        if best_valid >= err_v then
            best_valid = err_v
            best_train = err_r
            best_test = err_e
            best_epoch = t
        end
        print(string.format('Optima achieved at epoch %d: test: %.4f, valid: %.4f',
                            best_epoch, best_test, best_valid))
    end
    return report
end

local function optimConfig(opt)
    opt.optim_config = {
        learningRate      = opt.learning_rate,
        learningRateDecay = opt.decay_lr,
        weightDecay       = opt.l2reg,
        momentum          = opt.momentum
    }
    opt.optimizer = optim.sgd
end
-------------------------------------------------------------------------------
--                           Main Function                                   --
-------------------------------------------------------------------------------

local function main()
    local opt = commandLine()
    torch.setdefaulttensortype('torch.FloatTensor')
    local data = loadData()
    optimConfig(opt)
    
    local model, criterion = createModel(opt)
    model:cuda()
    criterion:cuda()
    print('model ')
    print(model)

    local confusion = optim.ConfusionMatrix(1)
    local W, grad = model:getParameters()

    print('the number of parameters is ' .. W:nElement())

    local report = reportErr(data, model, opt, confusion)
    
    for t = 1, opt.max_epoch do
        train(model, criterion, W, grad, data, opt)
        report(t)

        collectgarbage()
    end
end

main()

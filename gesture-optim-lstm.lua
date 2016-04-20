require 'rnn'
require 'optim'
require 'cunn'
dofile('loadGesture.lua')
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
    --cmd:option('-rho', 4, 'RNN steps')
    cmd:option('-rho', 5, 'RNN steps')
    cmd:option('-hidden_size', 90, 'hidden layer size')
    cmd:option('-input_size', 60, 'input size')
    cmd:option('-output_size', 8, 'output size')
    cmd:option('-learning_rate', 0.1, 'learning rate at t=0') 
    cmd:option('-momentum', 0.9, 'momentum (SGD only)')
    cmd:option('-decay_lr', 1e-4, 'learning rate decay')
    cmd:text()

    local opt = cmd:parse(arg)
    torch.manualSeed(opt.seed)
    return opt
end

local function predResult(val)
    res = val:clone()
    res[val:le(0)] = -1
    res[val:ge(0)] = 1
    return res
end

local function predResult(val)
    res = val:clone()
    res[val:le(0)] = -1
    res[val:ge(0)] = 1
    return res
end

-- build simple recurrent neural network
local function createModel(opt)
    local input_size, hidden_size, output_size = opt.input_size, opt.hidden_size, opt.output_size
    local rnn = nn.Sequential()
            :add(nn.Linear(input_size, hidden_size))
            :add(nn.LSTM(hidden_size, hidden_size))
            :add(nn.LSTM(hidden_size, hidden_size))
            :add(nn.Linear(hidden_size, output_size))
            :add(nn.LogSoftMax())

    rnn = nn.Sequencer(rnn)
    -- build criterion
    local criterion = nn.ClassNLLCriterion()
    local seqC = nn.SequencerCriterion(criterion)

    return rnn, seqC
end

-- build dummy dataset (task is to predict next item, given previous)
local function loadData()
    --local xr, yr= loadUCR('ItalyPowerDemand_TRAIN')
    local xr, yr= loadGesture('trainset.txt')
    --local xe, ye= loadUCR('ItalyPowerDemand_TEST')
    local xe, ye= loadGesture('testset.txt')

    local data = {}
    data['xr'] = xr
    data['yr'] = yr
    data['xe'] = xe
    data['ye'] = ye

    return data
end

-- x:cuda()
-- y:cuda()

-- training

function train(model, criterion, W, grad, data, opt)
    model:training()
    model:forget()

    local n_train = #data.xr


    for idx_ts = 1, n_train do
        local inputs, targets = {},{}
        for step = 1, data.xr[idx_ts]:size(1)/60 do
            inputs[step] = data.xr[idx_ts][{{opt.input_size * (step -1) + 1, opt.input_size * step}}]:cuda()
            targets[step] = data.yr[idx_ts]
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
    end
        
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
    local N = #data[name_x]
    local err = 0

    input = data[name_x]
    local datasets = {}
    
    for ind_ts = 1 , #input do 
        local inputs, targets = {}, data[name_y][ind_ts]
        local len_ts = input[ind_ts]:size(1)
        for step = 1, len_ts/60 do 
            inputs[step] = input[ind_ts][{{opt.input_size * (step -1) + 1, opt.input_size * step}}]:cuda()
        end

        local outputs = model:forward(inputs)
        local pred = outputs[len_ts/60]
        --pred = predResult(pred:resize(n_pred))

        confusion:add(pred, targets)

        confusion:updateValids()
        -- print(confusion)

        err = err + 1 - confusion.totalValid
        -- confusion:zero()
        --
        --for i = 1, n_pred do
        --    err = err + math.abs(pred[i] - targets[i])/2
        --end
    end
    return err/#input
end

local function reportErr(data, model, opt, confusion)
    local best_valid = math.huge
    local best_test = math.huge
    local best_train = math.huge
    local best_epoch = math.huge

    local function report(t)
        local err_r = evaluation('r', data, model, opt, confusion)
        -- local err_v = evaluation('v', data, model, opt, confusion)
        err_v = 0
        local err_e = evaluation('e', data, model, opt, confusion)
        print('---------Eopch: ' .. t .. ' of ' .. opt.max_epoch)
        print(string.format('Current Errors: train: %.4f | valid: %.4f | test: %.4f',
                            err_r, err_v, err_e))
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

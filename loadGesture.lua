require 'csvigo'

function loadGesture(file_name)
    f_train = csvigo.load({path = './gesture/' .. file_name, verbose = false, mode='raw'})

    train_set = {}
    -- load data set
    print ('loading ' .. file_name .. '...')
    for k,v in ipairs(f_train) do
        table.insert(train_set, torch.Tensor(string.split(v[1], ' +')))
    end

    train_targets = {}
    for i = 1, #train_set do
        table.insert(train_targets, train_set[i][1]) -- start from zero
        len_train = train_set[i]:size(1)
        train_set[i] = train_set[i]:sub(2, len_train)
    end

--    n_train = #train_set
--    len_train = train_set[1]:size(1)
--    datasets = torch.Tensor(n_train, len_train)
--    for i = 1, n_train do
--        datasets[{i,{}}] = train_set[i]:resize(1,len_train)
--    end
--
    return train_set, torch.Tensor(train_targets)
end


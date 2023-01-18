function [out] = group_test_case(dataLog,modelName)
    
    out = cell(numel(dataLog),1);
    for i = 1: numel(dataLog)
        if(contains(dataLog{i}.name, modelName))
            out{i} = dataLog{i};
        end
    end
    out = out(~cellfun('isempty',out));
end


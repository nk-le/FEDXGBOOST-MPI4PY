function out = Parser(file)
    fileHandler = regexp(fileread(file),'\n','split');
    out.param.xgbParam = getParam(fileHandler, "XGBoostParameter");
    out.param.quantileParam = getParam(fileHandler, "QuantileParameter");
    out.param.dataDistributionParam = getParam(fileHandler, "DataDistribution");

    out.quality = getParam(fileHandler, "Metrics");
    out.boosting = getParam(fileHandler, "Boosting");
    out.comm.Rx = getParam(fileHandler, "CommunicationRX");
    out.comm.Tx = getParam(fileHandler, "CommunicationTX");
    out.timing.fit = getParam(fileHandler, "FitTime");
    out.treeStructure = getParam(fileHandler, "TreeStructure");
  

    out.nTreeSetup = out.param.xgbParam .nTree;
    out.nTreeTrained = numel(out.treeStructure);
    out.nFeature = out.param.dataDistributionParam.nFeature;
    out.nUsers = out.param.dataDistributionParam.nTrain;
    % Compute the average time building a tree
    if(out.nTreeTrained > 1)
        dtArr = zeros(numel(out.timing.fit), 1);   
        for i = 1:numel(out.timing.fit)
            dtArr(i) = out.timing.fit{i}.dt;
        end
        out.dtTree = mean(dtArr);
    else
        out.dtTree = out.timing.fit.dt;
    end
    
    % Compute the total communication bytes
    nBytes = 0;
    for i = 1:numel(out.comm.Rx)
        if(out.comm.Rx{i}.TreeID == 0)
            nBytes = nBytes + out.comm.Rx{i}.nRx;
        end
    end

    for i = 1:numel(out.comm.Tx)
        if(out.comm.Tx{i}.TreeID == 0)
            nBytes = nBytes + out.comm.Tx{i}.nTx;
        end
    end
    out.nBytes = nBytes;
end


function outStruct = getParam(fileHandler, key)
    tmp = parseParam(fileHandler, key);
    if(numel(tmp) == 1)
        outStruct = tmp{:};
    else
        outStruct = tmp;
    end
end

function ret = parseParam(fileHandler, key)
    id = find(contains(fileHandler,key));
    orgStr = fileHandler(id);
    ret = cell(numel(orgStr),1);
    for j = 1:numel(ret)
        str = orgStr{j};
        sId = strfind(str,key);
        str = str(sId:end);
        infoArr = split(str,',');
        infoArr = strtrim(infoArr);
        out = struct();
        for i = 1:numel(infoArr)
            try
                tmp = split(infoArr{i},':');
                out.(tmp{1}) = str2double(tmp{2});
                
            catch
                
            end
        end
        ret{j} = out;
    end
end

function out = getQuantileParameter(fileHandler)
    id = find(contains(fileHandler,"QuantileParameter"));
    tmp = fileHandler(id);
    str = tmp{:};
    sId = strfind(str,'QuantileParameter');
    str = str(sId:end);
    infoArr = split(str,',');
    infoArr = strtrim(infoArr);
    out = struct();
    for i = 1:numel(infoArr)
        try
            tmp = split(infoArr{i},':');
            out.(tmp{1}) = str2double(tmp{2});
        catch

        end
    end
end

function dataLog = parse_folder(path)
    folders = dir(path);
    
    nExp = numel(folders);
    % Parsed all logfile into the dataLog handle
    dataLog = cell(nExp,1);
    for i = 1:nExp
    if(folders(i).name ~= "." && (folders(i).name ~= ".."))
        try
            tmp = Parser(fullfile(path, folders(i).name, "Rank_1.log"));
            tmp.name = folders(i).name;
            dataLog{i} = tmp;
        catch
            disp("Problem parsing: ");
            disp(folders(i).name);
        end
    end
    end
    
    dataLog = dataLog(~cellfun('isempty',dataLog));


end
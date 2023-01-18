function [out] = get_communication_bytes(handle)
    nTest = numel(handle);
    nBytes = zeros(nTest, 1);
    dt = zeros(nTest, 1);
    nUsers = zeros(nTest, 1);

    for i = 1: nTest
        nBytes(i) = handle{i}.nByte;
        dt(i) = handle{i}.dtTree;
        nUsers(i) = handle{i}.nUsers;
    end
    
    % Sorting for plotting
    [nUsers, sortIdx] = sort(nUsers,'ascend');
    % sort B using the sorting index
    nBytes = nBytes(sortIdx);


    out.nUsers = nUsers;
    out.nBytes = nBytes;
end


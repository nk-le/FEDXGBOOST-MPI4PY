function [out] = get_nUser_Time(inputArg1,inputArg2)
    nTest = numel(handle);
    nFeature = zeros(nTest, 1);
    dt = zeros(nTest, 1);
    nUsers = zeros(nTest, 1);
    for i = 1: nTest
        nFeature(i) = handle{i}.nFeature;
        dt(i) = handle{i}.dtTree;
        nUsers(i) = handle{i}.nUsers;
    end

    % Sorting for plotting
    [nFeature, sortIdx] = sort(nFeature,'ascend');
    % sort B using the sorting index
    dt = dt(sortIdx);


%     figure;
%     plot(nFeature, dt, '-o');
%     xlabel("# Features");
%     ylabel("Average iteration time [s]");
%     axis square;
% 
%     FormatFigure(gcf, 8/6, 4/3, "MarkerSize", 8);

    out.dtArr = dt;
    out.nFeatureArr = nFeature;
end


function [] = plotLoss(data)
    lossArr = zeros(data.nTree,1);
    for i = 1:data.nTree
        lossArr(i) = data.boosting{i}.Loss;
    end
    lossArr = lossArr/data.param.dataDistributionParam.nTest;
    lossArr = lossArr/norm(lossArr);
    figure
    plot(lossArr, '-o');
    xlabel("Iteration");
    ylabel("|L|");
    ylim([0, 1]);
    title("Loss Trajectory Evaluation")
    FormatFigure(gcf, 12/8, 4/3, "MarkerSize", 3);
end


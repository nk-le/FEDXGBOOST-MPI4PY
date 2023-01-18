%% Plot Feature vs Time
[nFeatureXgb, sortIdXGB] = sort(info.xgb.nFeature,'ascend');
[nFeatureFedXgb, sortIdFedXgb] = sort(info.fedxgb.nFeature,'ascend');

nFeatureXgb = [0; nFeatureXgb];
dtXgb = [0; info.xgb.dt(sortIdXGB)];
nFeatureFedXgb = [0; nFeatureFedXgb];
dtFedXgb = [0; info.fedxgb.dt(sortIdFedXgb)];

figure
plot(nFeatureXgb, dtXgb, '-o');
hold on; grid on; grid minor;
plot(nFeatureFedXgb, dtFedXgb, '-o');
FormatFigure(gcf, 12, 8/6, "MarkerSize", 20);
legend(["XGBoost", "FedXGBoost"], "Location", "northwest");
xlabel("# Features");
ylabel("Average iteration time [s]");
FormatFigure(gcf, 12, 12/8, "MarkerSize", 15);
title("Scalability Evaluation");
pbaspect([4 3 4]);
%xticks([10000:10000:40000]);
exportgraphics(gca,'ScalabilityFeature.eps','Resolution',1000); 
exportgraphics(gca,'ScalabilityFeature.png','Resolution',1000); 

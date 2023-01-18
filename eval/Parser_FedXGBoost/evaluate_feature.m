
figure
plot(sort([0; info.plainXGB.nFeature]) , sort([0; info.plainXGB.dt]), '-o', "DisplayName", "Vanilla XGBoost");
hold on; grid on; grid minor;
plot(sort([0; info.fedXGBSMM.nFeature]), sort([0; info.fedXGBSMM.dt]), '-o', "DisplayName", "FL XGBoost SMM");
%plot([0; info.fedXGBFSR.nUsers], [0; info.fedXGBFSR.dt], '-o', "DisplayName", "FL XGBoost SMM");
plot(sort([0; info.fedXGBFSR_0_03.nFeature]), sort([0; info.fedXGBFSR_0_03.dt]), '-o', "DisplayName", "FedXGBoost, r/n = 0.03");
plot(sort([0; info.fedXGBFSR_0_04.nFeature]), sort([0; info.fedXGBFSR_0_04.dt]), '-o', "DisplayName", "FedXGBoost, r/n = 0.04");
legend("Location", "northwest", "Interpreter", 'none');
xlabel("# Instances");
ylabel("Average Iteration Time [s]");
title("Feature Scalability Evalutation");
fig = gcf;
%fig.WindowState = 'maximized';
pbaspect([4 3 4]);
xticks([1e3:500:4e3]);
FormatFigure(gcf, 12, 12/8, "MarkerSize", 15);
exportgraphics(gca,'featureScale.eps','Resolution',1000); 
exportgraphics(gca,'featureScale.png','Resolution',1000); 

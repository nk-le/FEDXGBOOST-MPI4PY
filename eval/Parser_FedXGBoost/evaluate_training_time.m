
figure
plot([0; info.plainXGB.nUsers] , [0; info.plainXGB.dt], '-o', "DisplayName", "Vanilla XGBoost");
hold on; grid on; grid minor;
plot([0; info.fedXGB.nUsers], [0; info.fedXGB.dt], '-o', "DisplayName", "FL XGBoost SMM");
%plot([0; info.fedXGBFSR.nUsers], [0; info.fedXGBFSR.dt], '-o', "DisplayName", "FL XGBoost SMM");
plot(sort([0; info.fedXGBFSR_0_03.nUsers]), sort([0; info.fedXGBFSR_0_03.dt]), '-o', "DisplayName", "FedXGBoost, r/n = 0.03");
plot(sort([0; info.fedXGBFSR_0_04.nUsers]), sort([0; info.fedXGBFSR_0_04.dt]), '-o', "DisplayName", "FedXGBoost, r/n = 0.04");
legend("Location", "northeast", "Interpreter", 'none');
xlabel("# Instances");
ylabel("Average Iteration Time [s]");
title("Boosting Time Evalutation");
fig = gcf;
%fig.WindowState = 'maximized';
pbaspect([4 3 4]);
xticks([10000:10000:80000]);
FormatFigure(gcf, 12, 12/8, "MarkerSize", 15);
exportgraphics(gca,'timeEval.eps','Resolution',1000); 
exportgraphics(gca,'timeEval.png','Resolution',1000); 

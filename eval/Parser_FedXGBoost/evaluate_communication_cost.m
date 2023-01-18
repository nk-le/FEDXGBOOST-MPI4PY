
figure
% plot([0; info.plainXGB.nUsers] , [0; info.plainXGB.nBytes], '-o', "DisplayName", "Vanilla XGBoost");
% hold on; grid on; grid minor;
% plot([0; info.fedXGBFSR.nUsers], [0; info.fedXGBFSR.nBytes], '-o');
% plot([0; info.secureBoost.nUsers], [0; info.secureBoost.nBytes], '-o', "DisplayName", "SecureBoost");
% plot(sort([0; info.fedXGBFSR_0_03.nUsers]), sort([0; info.fedXGBFSR_0_03.nBytes]), '-o', "DisplayName", "FedXGBoost, r/n = 0.03");
% plot(sort([0; info.fedXGBFSR_0_04.nUsers]), sort([0; info.fedXGBFSR_0_04.nBytes]), '-o', "DisplayName", "FedXGBoost, r/n = 0.04");

semilogy([0; info.plainXGB.nUsers] , [0; info.plainXGB.nBytes], '-o', "DisplayName", "Vanilla XGBoost");
hold on; grid on; grid minor;
%plot([0; info.fedXGBFSR.nUsers], [0; info.fedXGBFSR.nBytes], '-o');
semilogy([0; info.secureBoost.nUsers], [0; info.secureBoost.nBytes], '-o', "DisplayName", "SecureBoost");
semilogy(sort([0; info.fedXGBFSR_0_03.nUsers]), sort([0; info.fedXGBFSR_0_03.nBytes]), '-o', "DisplayName", "FedXGBoost, r/n = 0.03");
semilogy(sort([0; info.fedXGBFSR_0_04.nUsers]), sort([0; info.fedXGBFSR_0_04.nBytes]), '-o', "DisplayName", "FedXGBoost, r/n = 0.04");


legend("Location", "east", "Interpreter", 'none');
xlabel("# Instances");
ylabel("# Bytes");
title("Communication Cost Evalutation");
fig = gcf;
%fig.WindowState = 'maximized';
pbaspect([4 3 4]);
xticks([10000:10000:80000]);
FormatFigure(gcf, 12, 12/8, "MarkerSize", 15);
exportgraphics(gca,'commEval.eps','Resolution',1000); 
exportgraphics(gca,'commEval.png','Resolution',1000); 

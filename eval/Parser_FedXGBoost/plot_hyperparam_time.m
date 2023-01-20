
[sGamma, sI] = sort(info.fedXGBFSR_0_03.gamma);
sDtFSR003 = info.fedXGBFSR_0_03.dt(sI);

[sGammaPlainXGB, sIPlainXGB] = sort(info.plainXGB.gamma);
sDtPlain = info.plainXGB.dt(sIPlainXGB);

sGamma = categorical(sGamma);
sGammaPlainXGB = categorical(sGammaPlainXGB);

figure
bar(sGamma, [sDtPlain, sDtFSR003]);
%bar(sGammaPlainXGB, sDtPlain, "DisplayName", "Vanilla XGBoost");
hold on; grid on; grid minor;
%bar(sGamma, sDtFSR003, "DisplayName", "FedXGBoost, r/n = 0.03");
%bar(sGammaPlainXGB, sDtPlain, "DisplayName", "Vanilla XGBoost");
legend(["Vanilla XGBoost", "FedXGBoost, r/n = 0.03"],"Location", "northeast", "Interpreter", 'none');
xlabel("Gamma");
ylabel("Average Iteration Time [s]");
title("Boosting Time Evalutation");
fig = gcf;
%fig.WindowState = 'maximized';
pbaspect([4 3 4]);
%xticks(sGamma);
FormatFigure(gcf, 12, 12/8, "MarkerSize", 15);
%exportgraphics(gca,'timeEval.eps','Resolution',1000); 
%exportgraphics(gca,'timeEval.png','Resolution',1000); 


%path = fullfile(pwd, "Log", "Test_NFEATURE_AugData");
%path = fullfile(pwd, "Log", "SCALABILITY_NINSTANCES_GivemeCredits");
path = fullfile(pwd, "Log", "Test_NUSERS_GivemeCredits");
%path = fullfile(pwd, "Log", "Test_NFEATURE_AugData/");
dataLogScalability.FedXGBoost = parse_folder(fullfile(path, "FedXGBoost"));
dataLogScalability.PlainXGBoost = parse_folder(fullfile(path, "PlainXGBoost"));
dataLogScalability.FedXGBoostFast = parse_folder(fullfile(path, "FedXGBoostFast"));
dataLogScalability.FedXGBoostFSR = parse_folder(fullfile(path, "FedXGBoostFSR"));
dataLogScalability.FedXGBoostR = parse_folder(fullfile(path, "FedXGBoostR"));
dataLogScalability.SecureBoost = parse_folder(fullfile(path, "SecureBoost"));
dataLogScalability.FedXGBoostFSR_0_03 = parse_folder(fullfile(path, "FedXGBoostFRS_R0_03"));
dataLogScalability.FedXGBoostFSR_0_04 = parse_folder(fullfile(path, "FedXGBoostFRS_R0_04"));
dataLogScalability.FedXGBoostSMM = parse_folder(fullfile(path, "FedXGBoostSMM"));

%handle.PlainXGBoost = group_test_case(dataLogScalability,"PlainXGBoost");
%handle.FedXGBoost = group_test_case(dataLogScalability,"FedXGBoost");
%handle.FedXGBoostFast = group_test_case(dataLogScalability,"FedXGBoostFast");

%handle.SecureBoost = group_test_case(dataLogScalability,"PseudoSecureBoost");

info.plainXGB = get_compare_info(dataLogScalability.PlainXGBoost);
info.fedXGB = get_compare_info(dataLogScalability.FedXGBoost);
info.fedXGBFast = get_compare_info(dataLogScalability.FedXGBoostFast);
info.fedXGBFSR = get_compare_info(dataLogScalability.FedXGBoostFSR);
info.fedXGBR = get_compare_info(dataLogScalability.FedXGBoostR);
info.secureBoost = get_compare_info(dataLogScalability.SecureBoost);
info.fedXGBFSR_0_03 = get_compare_info(dataLogScalability.FedXGBoostFSR_0_03);
info.fedXGBFSR_0_04 = get_compare_info(dataLogScalability.FedXGBoostFSR_0_04);
info.fedXGBSMM = get_compare_info(dataLogScalability.FedXGBoostSMM);





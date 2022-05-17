function [C,W,L] = nystrom_wrapper(X)
    rng(13)
    % define function for computing kernel dot product
    normc_fcn = @(m) sqrt(m.^2 ./ sum(m.^2));
    X = normc_fcn(X);

    gamma = min(size(X,1), 20);
    kFunc = @(X,rowInd,colInd) gaussianKernel(X,rowInd,colInd,gamma); % gaussianKernel

    % compute factors of Nyström approximation
    [C,W] = recursiveNystrom(X,gamma,kFunc ,0); % C*W*C'
    
    try
        L = chol(W);
    catch ME
        disp(ME.message)
        disp(W)
    end
end

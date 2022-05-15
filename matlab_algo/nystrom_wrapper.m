function [C, W] = nystrom_wrapper(X)
    % generate random test matrix
    %X = randn(2000,100); %X = normc(X);

    % define function for computing kernel dot product
    gamma = 3;
    kFunc = @(X,rowInd,colInd) gaussianKernel(X,rowInd,colInd,gamma);

    % compute factors of Nyström approximation
    [C,W] = recursiveNystrom(X,gamma,kFunc);
    
    %disp("X")
    %disp(X)
    %disp("CWC")
    %disp(C*W*C')


end

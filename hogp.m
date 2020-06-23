% multi-fidelity stochastic collation      
% 
% logg: v01 uses no low-fidelity groundtruth data
%       v02: funciton can take availabe low-fidelity data as inputs 
% 
% Inputs:
% xtr - [N_train x dim_x] matrix, input parameters
% ytr - [1 x m1 x m2 ... mk] cell, each element contains the corresponding output to
%       xtr and has to be a [N_train x dim_y] tensor. (output already tensorized)
% xte - [N_test x dim_x] matrix, testing inputs 
% rank - [1] # of latent featurs
% Kernel - [1 x 2] cell indicating the kernel used for x and for outputs  
% 
% Outputs:
% model
% 

function model = hogp(xTr, yTr, xTe, rank, Kernel)
    %% default using normalization for x and y
    nSample_tr = size(xTr,1);
    nSample_te = size(xTe,1);
    % dimX = 

    %for x 
    %seperate each dim
        meanX = mean(xTr);
        stdX = std(xTr);
    %combine each dim
    %             meanX = repmat(mean(xTr(:)), 1,size(xTr,2));
    %             stdX = repmat(std(xTr(:)), 1,size(xTr,2));
    %normalize data
    xTr = (xTr - repmat(meanX, nSample_tr, 1) ) ./ (repmat(stdX, nSample_tr, 1) + eps);
    xTe = (xTe - repmat(meanX, nSample_te, 1) ) ./ (repmat(stdX, nSample_te, 1) + eps);

    %for y 
    %seperate each dim 
    %     meanY = mean(yTr);
    %     stdY = std(yTr);
    %combine each dim
%                 meanY = repmat(mean(yTr(:)), 1,size(yTr,2));
%                 stdY = repmat(std(yTr(:)), 1,size(yTr,2));
                meanY = mean(yTr(:));
                stdY = std(yTr(:));
    %normalize data(
    yTr = (yTr - meanY ) ./ (stdY + eps);


    %% tensorize yTr
    yTr = tensor(yTr);  %require tensor package

%%
    if nargin < 5
        Kernel{1} = 'ard';
        Kernel{2} = 'ard';  
    end
    
    a0 = 1e-3;  %gamma prior for bta, typically 10^-3
    b0 = 1e-3;

    nvec = size(yTr);
    nmod = length(nvec);
    [~,d] = size(xTr);
    r = [d, rank*ones(1, nmod-1)];  %default low-rank for each dimension
%     r = ones(1, nmod)*rank;

    % MLE optimization setting     
    opt = [];
    opt.MaxIter = 300;
    opt.MaxFunEvals = 10000;


    %latent feature initialization
    U = cell(nmod, 1);
    U{1} = xTr;
    for k=2:nmod
        coor = linspace(-1,1,nvec(k));
        coor = (coor - mean(coor))/std(coor);
        %init with coordiates
        U{k} = repmat(coor', 1, r(k));
        %init with coordiates + small randomness
%         U{k} = U{k} + 0.0001*randn(size(U{k}));
        %random init
        %U{k} = randn(nvec(k), r(k));
%         U{k} = rand(nvec(k), r(k));
    end
    %init with Tucker
    %P = tucker_als_m1Fixed(ytr,r,Xtr);
    %U = P.U;
    %d = r(1);
    
    %init noise level
    log_bta = log(1/var(yTr(:)));
    params = [];
    
    for k=1:nmod
        if k>1
            params = [params;U{k}(:)];
        end
        log_l = zeros(r(k),1);
        %if k>1
        %    log_l = 2*log(median(pdist(U{k})))*ones(r(k),1);
        %end
        log_sigma0 = log(1e-4);
        %log_sigma0 = log(1);
        log_sigma = 0;
        params = [params;log_l;log_sigma;log_sigma0];
        
        %for ard-linear
        %log_alpha = 0;
        %if k>1
        %    params = [params;log_alpha];
        %end
    end
    params = [params;log_bta];
    %gradient check, no problem
    fastDerivativeCheck(@(params) log_evidence(params,r,a0,b0, xTr, yTr, Kernel{1}, Kernel{2}), params);
    new_params = minFunc(@(params) log_evidence(params, r, a0, b0, xTr, yTr, Kernel{1}, Kernel{2}), params,opt);
    %gradient check, no problem
    fastDerivativeCheck(@(params) log_evidence(params,r,a0,b0, xTr, yTr, Kernel{1}, Kernel{2}), params);
    
    %lbfgsb
%     opt = [];
%     funcl = @(params) log_evidence(params, r, a0, b0, Xtr, ytr, 'ard', 'ard');
%     l = -inf * ones(numel(params), 1);
%     u = inf*ones(numel(params), 1);
%     u(end) = log(1000);
%     opt.x0 = params;
%     opt.maxIts = 300;
%     [new_params, ~, ~] = lbfgsb(funcl, l, u, opt);
    
    
%     opt = [];
%     funcl = @(params) log_evidence(params, r, a0, b0, Xtr, ytr, 'ard', 'linear');
%     l = -inf * ones(numel(params), 1);
%     u = inf*ones(numel(params), 1);
%     u(end) = log(10);
%     opt.x0 = params;
%     opt.maxIts = 100;
%     [new_params, ~, ~] = lbfgsb(funcl, l, u, opt);
    for i = 1:length(xTe)
%         [yPred,model, yPred_tr] = pred_HoGP(new_params, r, Xtr, ytr, Xtest, 'ard', 'ard');
        [yPred,model, yPred_tr] = pred_HoGP_v2(new_params, r, xTr, yTr, xTe, 'ard', 'ard');
%         [yPred, pred_var, model, yPred_tr] = pred_HoGP_with_var(new_params, r, Xtr, ytr, Xtest, 'ard', 'linear');
    end
    
    %pred_mean = pred(params, r, Xtr, ytr, Xtest);
%     model.yPred = yPred;
%     model.yPred_tr = yPred_tr;
%     model.pred_var = pred_var;



    %de-normalize
    yPred = yPred .* stdY + meanY;       
%     model.yTe_pred = yTe_pred;
    yPred_tr = yPred_tr .* stdY + meanY;  
    fTe_var = model.var.data .*  stdY.^2;
%         yTe_var = repmat(yTe_var,1,size(yTr,2)) .* repmat(stdY,size(yTe_pred,1),1);
    

    model.yTe_pred = yPred.data;
    model.fTe_var = fTe_var;
    model.yTr_pred = yPred_tr.data;

end

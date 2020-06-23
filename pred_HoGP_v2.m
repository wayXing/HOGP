function [mu, model, mu_train] = pred_HoGP_v2(params, r, X, Y, Xtest, ker_type1, ker_type2)
% _v2: add prediction of variance

    [N,d] = size(X);
    nvec = size(Y); %it must be [N, m1, m2, ..., mK]
    assert(N==nvec(1), 'inconsistent input-output');
    nmod = length(nvec);
    %each dimension has a diffrent kernel, let us assume ARD first
    ker_params = cell(nmod,1);
    U = cell(nmod, 1);
    U{1} = X;
    %extract parameters:    
    [ker_params{1},idx] = load_kernel_parameter(params,d, ker_type1, 0);
    for k=2:nmod
        U{k} = reshape(params(idx+1: idx+nvec(k)*r(k)),nvec(k), r(k));
        [ker_params{k},idx] = load_kernel_parameter(params, r(k), ker_type2, idx+nvec(k)*r(k));
    end
    bta = exp(params(idx+1));
    Sigma = cell(nmod, 1);Lam = cell(nmod, 1);LamDiag = cell(nmod,1);V = cell(nmod,1);Vt = cell(nmod,1);
    for k=1:nmod
        Sigma{k} = ker_func(U{k}, ker_params{k});
        [V{k}, LamDiag{k}] = eig(Sigma{k});
        Lam{k} = diag(LamDiag{k});
        Vt{k} = V{k}';
    end
    btaInvPlusSigma = 1/bta + tensor(ktensor(Lam));
    %M = tenfun(@(x)(1./x), btaInvPlusSigma);
    M = 1./btaInvPlusSigma;
    T = ttm(times(M, ttm(Y,Vt)), V);
    mu_train = ttm(T, Sigma); 
    Sigma{1} = ker_cross(Xtest, X, ker_params{1});
    mu = ttm(T, Sigma);
    model = [];
    model.ker_params = ker_params;
    model.bta = bta;
    model.U = U;
    
    %var
%     covx = ker_cross(X, X, ker_params{1})
%     covx = ker_func(U{1}, ker_params{1});
%     inv_covx = inv(covx);
%     inv_covx2 = V{1} * diag(1./Lam{1}) * V{1}';
    
    S = times(tensor(ktensor(Lam)),M.^(1/2));
    S2 = S.^2;
    
    k_self{1} = (diag(ker_func(Xtest, ker_params{1})));
    for j = 2:length(nvec)
        k_self{j} = (diag(Sigma{j}));
    end
    var_part1 = tensor(ktensor(k_self));
    
    
    kstar = ker_cross(Xtest, X, ker_params{1});
    MS{1} = (kstar * V{1} * diag(1./Lam{1})).^2;     
    for j = 2:length(nvec)
        MS{j} = V{j}.^2;
    end
    var_part2 = ttm(S2, MS);
        
    var = var_part1 - var_part2;
    model.var = var;
    
    %var method 2
%     for i = 1:size(Xtest,1)  
%     for i = 1
%         xStar = Xtest(i,:);
%         Sigma{1} = ker_cross(xStar, X, ker_params{1});
%         for j = 1:length(V)
%             vtkt{j} =  Vt{j} * Sigma{j}';
%             vk{j} =  Sigma{j} * V{j} ;
%         end
%         
%         var_2 = ttm(ttm(M,vk),vtkt);
%         Sigma{1} = ker_cross(xStar, X, ker_params{1});
%         tempY = ttm(tensor(ktensor(tempLam)), Sigma);
%         T = ttm(times(M, ttm(Sigma, Vt)), V);
%         varTe = ttm(T, Sigma); 
%     end

end
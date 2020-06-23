%ensemble with full covairnace on q(w)
%param: a vector of parameters, inculding kernel paramters, noise inverse variance 
%a0, b0: gamma prior for bta, e.g., 10^-3
%X: training input, N by d matrix
%Y: training output, N by m1 by m2 ... by mK tensor (output already tensorized)
%r: rank setting, r(1) = d, r(2:K) is the rank of latent features
%output: logL: log model evidence, dLogL: gradient
function [f, df] = log_evidence(params, r, a0, b0, X, Y, ker_type1, ker_type2)
    [N,d] = size(X);
    nvec = size(Y); %it must be [N, m1, m2, ..., mK]
    assert(N==nvec(1), 'inconsistent input-output');
    nmod = length(nvec);
    %each dimension has a diffrent kernel, let us assume ARD first
    ker_params = cell(nmod,1);
    U = cell(nmod, 1);
    U{1} = X;
    %extract parameters
    [ker_params{1},idx] = load_kernel_parameter(params, d, ker_type1, 0);
    for k=2:nmod
        U{k} = reshape(params(idx+1:idx+nvec(k)*r(k)),nvec(k), r(k));
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
    logL = (a0 - 1)*log(bta) - b0*bta;
    dbta = (a0 - 1)/bta - b0;
    %log|bta^{-1} + \Sigma \kron ...|
    btaInvPlusSigma = 1/bta + tensor(ktensor(Lam));
    M = 1./btaInvPlusSigma;
    M12 = tenfun(@sqrt,M);
    logL = logL - 0.5*sum(log(btaInvPlusSigma(:)));
    %sum(btaInvPlusSigma(:)<0)
    dbta = dbta + 0.5*bta^(-2)*sum(M(:));
    dker_params = cell(nmod,1);
    dU = cell(nmod, 1);
    for k=1:nmod
        bk = ttv(M,Lam,setdiff(1:nmod,k));
        Ak = -0.5*V{k}*diag(bk.data)*V{k}';
        [dU{k}, dker_params{k}] =  ker_grad(U{k}, Ak, Sigma{k},ker_params{k});        
    end
    %-0.5vec(Y)^\top (bta^-1 I + \Sigma)^{-1} vec(Y)
    T = times(M12, ttm(Y, Vt));
    T2 = times(M12,T);
    D12Y = ttm(T, V);
    DY = ttm(T2, V);
    logL = logL - 0.5*sum(D12Y(:).*D12Y(:));
    dbta = dbta - 0.5*bta^(-2)*sum(DY(:).*DY(:));    
    %C is T2;
    for k=1:nmod
        Fk = ttm(T2,LamDiag,setdiff(1:nmod, k));
        Fks = tenmat(Fk, k);
        Cs = tenmat(T2, k);
        Ak = 0.5*(V{k}*(Cs.data*Fks.data')*Vt{k});
        [dUk, dkp] = ker_grad(U{k}, Ak, Sigma{k}, ker_params{k});
        dker_params{k} = dker_params{k} + dkp;
        if k>1
            dU{k} = dU{k} + dUk;    
        end
    end
    d_log_bta = bta*dbta;
    %assemble gradients
    %regularizaton
    logL = logL - 0.5*sum(params(1:end-1).*params(1:end-1));% - 0.5*bta*bta;
    %d_log_bta = d_log_bta - bta*bta;
    f = -logL;
    df = zeros(numel(params),1);
    idx = 0;
    df(idx+1:idx+length(dker_params{1})) = dker_params{1};
    idx = idx + length(dker_params{1});
    for k=2:nmod
        df(idx+1:idx+numel(dU{k})) = dU{k}(:);
        idx = idx + numel(dU{k});
        df(idx+1:idx+numel(dker_params{k})) = dker_params{k};
        idx = idx + numel(dker_params{k});
    end
    %if bta>10
    %    d_log_bta = 0;
    %end
    df(idx+1) = d_log_bta;  
    %regularizaton
    df(1:end-1) = df(1:end-1) - params(1:end-1);
    df = -df;
end
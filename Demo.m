% Demo
clear 

ntr = 16;
nte = 256;

y_index = 0.1:0.1:3;

xtr = rand(ntr,1);
for i = 1:ntr
    ytr(i,:) = xtr(i) .* sin(xtr(i) .* y_index) + rand(1,length(y_index))*0.05;
%     ytr(i,:) = xtr(i) .* sin(xtr(i) .* y_index);
end


xte = linspace(0,1,nte)';
for i = 1:nte
    yte(i,:) = xte(i) .* sin(xte(i) .* y_index);
end


model = hogp(xtr, ytr, xte, 6);
% model2 = hogp(xtr, ytr, xte, 6, {'ard','linear'});

%% plot
figure(1)
mesh(yte)
title('ground truth')

figure(2)
mesh(model.yTe_pred)
title('predictions')

% figure(3)
% mesh(model2.yTe_pred)


%% random testing 
% x = rand(64,1);
% y = reshape(sinc(x(:) * x(:)'),64,8,8);
% 
% xtr = x(1:24,:);
% ytr = y(1:24,:,:);
% 
% xte = x(25:end,:);
% yte = y(25:end,:,:);

% model = train_HOGP(xtr, tensor(ytr), xte, tensor(yte), 2, 0.001, 0.001);
% [pred_mean,model, pred_tr] = pred_HoGP(new_params, r, Xtr, ytr, Xtest, 'ard', 'linear');

% model = hogp(xtr, ytr, xte, 2);
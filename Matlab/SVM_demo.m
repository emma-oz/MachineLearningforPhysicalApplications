% % SVM in MATLAB
% % Emma Reeves. May 1, 2017
% % libsvm package can be found on github: https://github.com/cjlin1/libsvm
clear; clc;
%close all;
PLOT = 'on'; %(set PLOT == 'off' to suppress plotting)
Type =  'Classify'; %(set 'Classify' OR 'Regress')'Regress';

%% generate data (for the example case)
rng(3);
R = 1; %inner radius
r = 1.4; %outer radius
N = 2000;
X = (4.*rand(N,2) - 2);
Y = -1*ones(size(X,1),1);
%pos = find(((d(:,1)+1).^2 + d(:,2).^2)<1 | ((d(:,1)-2).^2 + d(:,2).^2)<1); %two circles
Y((X(:,1).^2 + X(:,2).^2)<=R.^2) = 1; % create middle circle
pos = find(abs((X(:,1).^2 + X(:,2).^2)>R.^2) & (abs(X(:,1).^2 + X(:,2).^2)<r.^2)); % remove the points between two circles
Y(pos) = [];
X(pos,:) = [];
X=X+1;
%% Train SVM model (Skip directly to this if you already have data)
% optional kernels to use
kernel_opts = {'-t 0','-t 2','-t 1','-t 3'};
titles = {'Linear Kernel','Radial Basis Function Kernel','Polynomial Kernel','Sigmoid Function Kernel'};
gamma = '0.1166';

kernel = kernel_opts{2};
    
    % svmtrain(Y, X, options). 
    % OPTIONS == 
        % -c #: cost function (use 1 by default)
        % -g #: gamma 
        % -q: quiet output to Command Window
        % -t #: 0 == linear, 1 == Poly, 2 == RBF, 3 == Sigmoid, 4 == custom
    
    switch Type
        case 'Classify'
            % train
                model = svmtrain(Y, X,['-c 7.46 -g ' gamma ' -q ' kernel]);
            % predict
                [predict_label,~, ~] = svmpredict(rand([length(Y),1]), X, model,'-q'); %use dummy label inputs

        case 'Regress'
            % train
                model = svmtrain(Y, X,...
                    ['-s 4 -n 0.5 -c 1 -g ' gamma ' -q ',kernel]);
            % predict
                [y_pred,~, ~] = svmpredict(rand([length(Y),1]), X, model,'-q'); % use dummy label inputs
    end

    
%% Plot results

% create data mesh for mapping
figure(1); hold on
x_map = linspace(min(X(:))-0.25,max(X(:))+0.25,floor(length(X)/10));
y_map = linspace(min(X(:))-0.25,max(X(:))+0.25,floor(length(X)/10));
[X_map,Y_map] = meshgrid(x_map,y_map); % create a mesh
boundary_map = [X_map(:), Y_map(:)];

% plot the decision boundary
boundary = svmpredict(rand([length(boundary_map),1]),boundary_map,model,'-q');
pcolor(X_map,Y_map,reshape(boundary,[length(X_map),length(Y_map)]));
shading flat;
colormap([0.8 0.8 1; 1 0.8 0.8]);

% plot the data
pos = find(Y==1);
scatter(X(pos,1),X(pos,2),10,'ro','filled'); 
pos = find(Y==-1);
scatter(X(pos, 1),X(pos, 2),10,'bo','filled');

% get and plot support vectors
sv = full(model.SVs); 
plot(sv(:,1),sv(:,2),'ko','linewidth',2,'MarkerSize',6);

xlim([min(X(:))-0.25 max(X(:))+0.25]);
ylim(xlim);
xlabel('x1'); ylabel('x2');
title(titles{strmatch(kernel,kernel_opts)});
    
    
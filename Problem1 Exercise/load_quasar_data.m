% load_quasar_data
%
% Loads the data in the quasar data files
%
% Upon completion of this script, the matrices and data are as follows:
%
% lambdas - A length n = 450 vector of wavelengths {1150, ..., 1599}
% train_qso - A size m-by-n matrix, where m = 200 and n = 450, of noisy
%      observed quasar spectra for training.
% test_qso - A size m-by-n matrix, where m = 200 and n = 450, of noisy observed
%       quasar spectra for testing.

load quasar_train.csv;
lambdas = quasar_train(1, :)';
train_qso = quasar_train(2:end, :);
load quasar_test.csv;
test_qso = quasar_test(2:end, :);

%fristTrian=train_qso(1, :);
%theta=((lambdas'*lambdas)^-1)*lambdas'*fristTrian'
%scatter(lambdas, fristTrian,'+')
[mm, nn] = size(train_qso);
mtest = size(test_qso, 1);
y = train_qso(1, :)';
X = [ones(nn, 1), lambdas];
theta = X\y; % Solves linear regression directly,wocao 
%theta=((X'*X)^-1)*X'*y'
figure;
h = plot(lambdas, train_qso(1, :), 'k+');
set(h, 'linewidth', 1);
hold on;
h = plot(lambdas, theta(1) + lambdas * theta(2), 'r-');
set(h, 'linewidth', 2);
h = legend('Raw data', 'Regression line');
set(h, 'fontsize', 20);
print -depsc2 quasar_linear_regression.eps;


%% Part (a)ii/iii: Smoothing the quasars with LWLR
figure;
X = [ones(nn, 1), lambdas];
y = train_qso(1, :)';
h = plot(lambdas, y, 'k+');
set(h, 'linewidth', 1);
hold on;
colors = {'r-', 'b-', 'g-', 'm-', 'c-'};
taus = [1, 5, 10, 100, 1000];
for tau_ind = 1:5
tau = taus(tau_ind);
y_smooth = local_linear_regression(lambdas, y, tau);
h = plot(lambdas, y_smooth, char(colors(tau_ind)));
set(h, 'linewidth', 2);
end
h = legend('Raw data', 'tau = 1', 'tau = 5', 'tau = 10', ...
'tau = 100', 'tau = 1000');
set(h, 'fontsize', 20);
print -depsc2 sprintf('quasar_locally_taus.eps', tau);

%% Part (b)i: Smooth all quasars with LWLR
train_smooth = train_qso;
test_smooth = test_qso;
tau = 5;
X = [ones(nn, 1), lambdas];
for jj = 1:mm
ytrain = train_qso(jj, :)';
train_smooth(jj, :) = local_linear_regression(lambdas, ytrain, tau)';
end
for jj = 1:mtest
ytest = test_qso(jj, :)';
test_smooth(jj, :) = local_linear_regression(lambdas, ytest, tau)';
end
%% Find the right-most function parts
right_trains = train_smooth(:, 151:end);
left_trains = train_smooth(:, 1:50);
right_tests = test_smooth(:, 151:end);
left_tests = test_smooth(:, 1:50);
%% Construct matrix of all pairs of distances between training quasar spectra
train_dists = zeros(mm, mm);
for ii = 1:mm
for jj = (ii + 1):mm
train_dists(ii, jj) = norm(right_trains(ii, :) - right_trains(jj, :))^2;
end
end
train_dists = train_dists + train_dists';
train_dists = train_dists / max(train_dists(:));
%% Reconstruct training curves
f_left_estimates = zeros(mm, 50);
num_nearest = 3;
for ii = 1:mm
[train_dist_sort, inds] = sort(train_dists(:, ii), 1, 'ascend');
close_inds = ones(mm, 1);
close_inds(inds((num_nearest + 1):end)) = 0;
h = max(train_dists(:, ii));
kerns = max(1 - train_dists(:, ii) / h, 0); % An m-by-1 vector
kerns = kerns .* close_inds;
f_left_estimates(ii, :) = left_trains' * kerns / sum(kerns);
end

% Compute error rate of estimates
err = sum((left_trains(:) - f_left_estimates(:)).^2);
err = err / mm;
fprintf(1, 'Average training error: %1.4f\n', err);
%% Reconstruct test curves
% Construct matrix of all pairs of distances between training and testing
% quasar spectra
train_to_test_dists = zeros(mm, mtest);
for ii = 1:mm
for jj = 1:mtest
train_to_test_dists(ii, jj) = ...
norm(right_trains(ii, :) - right_tests(jj, :))^2;
end
end
train_to_test_dists = train_to_test_dists / max(train_to_test_dists(:));
f_left_estimates = zeros(mtest, 50);
for ii = 1:mtest
[tttd_sorted, inds] = sort(train_to_test_dists(:, ii), 1, 'ascend');
close_inds = ones(mm, 1);
close_inds(inds((num_nearest + 1):end)) = 0;
h = max(train_to_test_dists(:, ii));
kerns = max(1 - train_to_test_dists(:, ii) / h, 0);
kerns = kerns .* close_inds;
f_left_estimates(ii, :) = left_trains' * kerns / sum(kerns);
end
%% Compute error rate of estimates
err = sum((left_tests(:) - f_left_estimates(:)).^2);
err = err / mtest;
fprintf(1, 'Average testing error: %1.4f\n', err);
%% Final plots
figure;
plot(lambdas, test_smooth(1, :), 'k-', 'linewidth', 1);
hold on;
plot(lambdas(1:50), f_left_estimates(1, :), 'r-', 'linewidth', 2);
print -depsc2 'quasar_test_1.eps';
figure;
plot(lambdas(1:50), f_left_estimates(6, :), 'r-', 'linewidth', 2)
hold on;
plot(lambdas, test_smooth(6, :), 'k-', 'linewidth', 1);
print -depsc2 'quasar_test_6.eps';
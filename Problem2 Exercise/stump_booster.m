function [theta, feature_inds, thresholds] = stump_booster(X, y, T)
% STUMP_BOOSTER Uses boosted decision stumps to train a classifier
%
% [theta, feature_inds, thresholds] = stump_booster(X, y, T)
%  performs T rounds of boosted decision stumps to classify the data X,
%  which is an m-by-n matrix of m training examples in dimension n,
%  to match y.
%
%  The returned parameters are theta, the parameter vector in T dimensions,
%  the feature_inds, which are indices of the features (a T dimensional
%  vector taking values in {1, 2, ..., n}), and thresholds, which are
%  real-valued thresholds. The resulting classifier may be computed on an
%  n-dimensional training example by
%
%   theta' * sign(x(feature_inds) - thresholds).
%
%  The resulting predictions may be computed simultaneously on an
%  n-dimensional dataset, represented as an m-by-n matrix X, by
%
%  sign(X(:, feature_inds) - repmat(thresholds', m, 1)) * theta.
%
%  This is an m-vector of the predicted margins.

[mm, nn] = size(X);
p_dist = ones(mm, 1);
p_dist = p_dist / sum(p_dist);

theta = [];
feature_inds = [];
thresholds = [];

for iter = 1:T
  [ind, thresh] = find_best_threshold(X, y, p_dist);
  Wplus = p_dist' * (sign(X(:, ind) - thresh) == y);
  Wminus = p_dist' * (sign(X(:, ind) - thresh) ~= y);
  theta = [theta; .5 * log(Wplus / Wminus)];
  feature_inds = [feature_inds; ind];
  thresholds = [thresholds; thresh];
  % ------- You should implement your code here -------- %
  p_dist = exp(-y .* (...
  sign(X(:, feature_inds) - repmat(thresholds', mm, 1)) * theta));
  fprintf(1, 'Iter %d, empirical risk = %1.4f, empirical error = %1.4f\n', ...
  iter, sum(p_dist), sum(p_dist >= 1));
  p_dist = p_dist / sum(p_dist);


  newest_theta_param = 0;  % Change this line so that newest_theta_param takes
                           % the optimal weight for the new decision stump.

  % ------- No need to change this part ------- %
  theta = [theta; newest_theta_param];
end

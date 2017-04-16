function [ind, thresh] = find_best_threshold(X, y, p_dist)
% FIND_BEST_THRESHOLD Finds the best threshold for the given data
%
% [ind, thresh] = find_best_threshold(X, y, p_dist) returns a threshold
%   thresh and index ind that gives the best thresholded classifier for the
%   weights p_dist on the training data. That is, the returned index ind
%   and threshold thresh minimize
%
%    sum_{i = 1}^m p(i) * 1{sign(X(i, ind) - thresh) ~= y(i)}
%
%   OR
%
%    sum_{i = 1}^m p(i) * 1{sign(thresh - X(i, ind)) ~= y(i)}.
%
%   We must check both signed directions, as it is possible that the best
%   decision stump (coordinate threshold classifier) is of the form
%   sign(threshold - x_j) rather than sign(x_j - threshold).
%
%   The data matrix X is of size m-by-n, where m is the training set size
%   and n is the dimension.
%
%   The solution version uses efficient sorting and data structures to perform
%   this calculation in time O(n m log(m)), where the size of the data matrix
%   X is m-by-n.

[mm, nn] = size(X);
ind = 1;
thresh = 0;

% ------- Your code here -------- %
%
% A few hints: you should loop over each of the nn features in the X
% matrix. It may be useful (for efficiency reasons, though this is not
% necessary) to sort each coordinate of X as you iterate through the
% features.
for jj = 1:nn
    [x_sort, inds] = sort(X(:, jj), 1, 'descend');
    p_sort = p_dist(inds);
    y_sort = y(inds);
    % We let the thresholds be s_0, s_1, ..., s_{m-1}, where s_k is between
    % x_sort(k-1) and x_sort(k) (so that s_0 > x_sort(1)). Then the empirical
    % error associated with threshold s_k is exactly
    %
    % err(k) = sum_{l = k + 1}^m p_sort(l) * 1(y_sort(l) == 1)
    % + sum_{l = 1}^k p_sort(l) * 1(y_sort(l) == -1),
    %
    % because this is where the thresholds fall. Then we can sequentially
    % compute
    %
    % err(l) = err(l - 1) - p_sort(l) y_sort(l),
    %
    % where err(0) = p_sort' * (y_sort ==1).
    % 表示threshold的index为1，大于index=1的都被分类为1
    %
    % The code below actually performs this calculation with indices shifted by
    % one due to Matlab indexing.
    s = x_sort(1) + 1;
    possible_thresholds = x_sort;
    possible_thresholds = (x_sort + circshift(x_sort, 1)) / 2;
    possible_thresholds(1) = x_sort(1) + 1;
    increments = circshift(p_sort .* y_sort, 1);
    increments(1) = 0;
    emp_errs = ones(mm, 1) * (p_sort' * (y_sort == 1));%index=1的情况
    emp_errs = emp_errs - cumsum(increments);%根据上面给出的公式，这里用cumsum代表递归
    [best_low, thresh_ind] = min(emp_errs);
    [best_high, thresh_high] = max(emp_errs);
    best_high = 1 - best_high;%之所以这样做是因为thresh - X互换位置，见Boosting讲义
    best_err_j = min(best_high, best_low);
    if (best_high < best_low)
        thresh_ind = thresh_high;
    end
    if (best_err_j < best_err)
        ind = jj;
        thresh = possible_thresholds(thresh_ind);
        best_err = best_err_j;
    end
end
% 举例：A = [ 3 3 5
%           0 4 2 ];
% sort(A,1) %纵向排列
% ans =
%      0     3     2
%      3     4     5
% sort(A,2) %横向排列
% ans =
%      3     3     5
%      0     2     4


% A = [ 1 2 3;4 5 6; 7 8 9];
%        B = circshift(A,1) % circularly shifts first dimension values down by 1.
%        B =     7     8     9
%                1     2     3
%                4     5     6
%        B = circshift(A,[1 -1]) % circularly shifts first dimension values
%                                % down by 1 and second dimension left by 1.
%        B =     8     9     7
%                2     3     1
%                5     6     4
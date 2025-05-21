% Create a mapping from old clusters to new clusters
new_idx = idx; % Copy the original cluster labels

% Define the mapping: (old → new)
new_idx(idx == 1) = 4;
new_idx(idx == 2) = 1;
new_idx(idx == 3) = 2;
new_idx(idx == 4) = 3;
% new_idx(idx == 5) = 4;
% new_idx(idx == 6) = 2;

% Now use 'new_idx' instead of 'idx' in your plots

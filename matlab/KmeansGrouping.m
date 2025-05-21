% Assuming 'data' is your 237x2 dataset
k = 4; % Number of clusters (since you want 4 classes)

% Step 1: Extract the second feature (column 2) from the data
%secondFeature = Final_AllMix(:, 2);
secondFeature = P_B2;

% Step 2: Perform k-means clustering on the second feature
[idx, centers] = kmeans(secondFeature, k);

% Step 3: Calculate the range (min and max) for each cluster based on the second feature
clusterRanges = zeros(k, 2); % Initialize a matrix to store [min, max] for the second feature

for i = 1:k
    % Extract the data points that belong to the current cluster
    clusterData = secondFeature(idx == i);
    
    % Calculate the range for the second feature in this cluster
    clusterRanges(i, :) = [min(clusterData), max(clusterData)];
end

% Step 4: Display the cluster centers and ranges based on the second feature
disp('Cluster Centers based on Second Feature:');
disp(centers);

disp('Cluster Ranges based on Second Feature:');
for i = 1:k
    fprintf('Cluster %d:\n', i);
    fprintf('  Second Feature range: [%.2f, %.2f]\n', clusterRanges(i, 1), clusterRanges(i, 2));
end

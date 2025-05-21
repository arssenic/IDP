PCPZ = P_B2;
scatter(PCPZ(:,1), PCPZ(:,2), 50, idx, 'filled'); % Scatter plot with cluster colors
hold on;
scatter(centers(:,1), centers(:,2), 100, 'kx', 'LineWidth', 2); % Mark centroids
hold off;

xlabel('\theta_{PC1}', 'FontSize', 20); 
ylabel('\theta_{PC2}', 'FontSize', 20); 

% Increase font size of tick labels
ax = gca; % Get current axes
ax.FontSize = 20; % Set font size of tick values

colorbar;
legend('Clustered Data Points', 'Cluster Centers', 'FontSize', 20);

grid on; 
axis equal; % Ensures equal scaling for better visualization

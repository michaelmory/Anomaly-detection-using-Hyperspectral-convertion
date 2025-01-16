% Load the similarity mask
load('similarity_mask.mat', 'similarity_mask'); % Load the similarity mask

% Visualize the similarity mask
figure;
imshow(similarity_mask, []);
title('Original Similarity Mask');

% Detect connected components in the mask
[mask_labels, num_regions] = bwlabel(similarity_mask);

% Analyze regions and detect anomalies based on size or compactness
min_region_area = 5; % Minimum area for a valid region (adjust as needed)
max_region_area = 300; % Maximum area for a valid region (adjust as needed)

% Initialize anomaly map
anomaly_mask = zeros(size(similarity_mask));

for i = 1:num_regions
    % Get the current region
    current_region = (mask_labels == i);
    
    % Compute region properties
    region_props = regionprops(current_region, 'Area', 'Eccentricity');
    
    % Check if the region falls outside valid criteria
    if region_props.Area < min_region_area || region_props.Area > max_region_area
        % Mark this region as an anomaly
        anomaly_mask = anomaly_mask | current_region;
    end
end

% Visualize the detected anomalies
figure;
imshow(anomaly_mask, []);
title('Anomalies in Similarity Mask');

% Highlight anomalies on the original mask
figure;
imshow(similarity_mask, []);
hold on;
[B, ~] = bwboundaries(anomaly_mask, 'noholes');
for k = 1:length(B)
    boundary = B{k};
    plot(boundary(:, 2), boundary(:, 1), 'r', 'LineWidth', 1.5);
end
hold off;
title('Detected Anomalies Highlighted in Red');

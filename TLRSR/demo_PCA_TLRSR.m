% Clear workspace
clear all;
close all;
clc;
addpath(genpath('./tSVD'));
addpath(genpath('./proximal_operator'));

% Load your hyperspectral image and annotations
load('./HS/1039.mat'); % Replace with the name of your .mat file
load('./GT/pos_G3P3371GT.mat'); % Replace with the name of your annotations file

data = rec_hs; % Hyperspectral image from your .mat file
% --- New: Load the average spectrum ---
load('average_spectrum.mat', 'average_spectrum'); % Precomputed from multiple images

% --- New: Generate similarity mask using the original data ---
[rows, cols, bands] = size(data);
distance_map = zeros(rows, cols);

for i = 1:rows
    for j = 1:cols
        current_spectrum = squeeze(data(i, j, :));
        distance_map(i, j) = norm(current_spectrum - average_spectrum);
    end
end

% Define a threshold based on the original data
similarity_threshold = 0.5; % Adjust this value as needed
similarity_mask = distance_map <= similarity_threshold;

% Save the similarity mask for later use
save('similarity_mask.mat', 'similarity_mask');

% Visualize the similarity mask
figure;
imagesc(similarity_mask);
colormap('gray');
colorbar;
title('Similarity Mask');
% Normalize the hyperspectral image
[H, W, Dim] = size(data);

for i = 1:Dim
    min_val = min(min(data(:, :, i)));
    max_val = max(max(data(:, :, i)));
    data(:, :, i) = (data(:, :, i) - min_val) / (max_val - min_val);
end
% Reshape the image data for PCA
X = reshape(data, [], Dim); % Shape: [H*W x Dim]

% Dimensionality reduction using PCA
numb_pca_components = 2; % Number of PCA components to retain
[coeff, score] = pca(X); % PCA transformation
reduced_X = score(:, 1:numb_pca_components); % Retain top PCA components

% Apply ICA on PCA-reduced data
num_ica_components = 20; % Number of ICA components to extract
[icasig, ~, ~] = fastica(reduced_X', 'numOfIC', num_ica_components);

% Check the actual number of components computed
actual_num_components = size(icasig, 1);
if actual_num_components < num_ica_components
    warning(['FastICA converged with fewer components than requested: ', ...
             num2str(actual_num_components), ' out of ', num2str(num_ica_components)]);
    num_ica_components = actual_num_components;
end

% Reshape ICA components back to spatial dimensions
ica_images = reshape(icasig', H, W, num_ica_components);

% Sliding window entropy-based anomaly detection
window_size = 31; % Sliding window size (must be odd)
if mod(window_size, 2) == 0
    window_size = window_size + 1; % Ensure window size is odd
end

entropy_scores = zeros(1, num_ica_components);
for i = 1:num_ica_components
    component = ica_images(:, :, i);
    
    % Normalize component to [0, 1]
    component = (component - min(component(:))) / (max(component(:)) - min(component(:)));
    
    % Calculate local entropy with sliding window
    local_entropy = entropyfilt(component, true(window_size));
    
    % Calculate the maximum entropy value within the component
    entropy_scores(i) = max(local_entropy(:));
end

% Select top components based on entropy scores
[~, top_entropy_indices] = sort(entropy_scores, 'descend');
top_components = top_entropy_indices(1:min(3, num_ica_components)); % Select top 3 components

% Visualization of selected components based on entropy
for i = 1:length(top_components)
    component = ica_images(:, :, top_components(i));
    local_entropy = entropyfilt((component - min(component(:))) / ...
                                 (max(component(:)) - min(component(:))), true(window_size));
    
    % Visualize the original component
    figure;
    subplot(1, 2, 1);
    imagesc(component);
    colormap('jet');
    colorbar;
    title(['Selected Component ', num2str(top_components(i)), ...
           ' (Entropy Score: ', num2str(entropy_scores(top_components(i))), ')']);
    axis image off;
    
    % Visualize the local entropy map
    subplot(1, 2, 2);
    imagesc(local_entropy);
    colormap('gray');
    colorbar;
    title(['Local Entropy Map - Component ', num2str(top_components(i))]);
    axis image off;
end

% Continue with anomaly detection and bounding box detection
% Update DataTest with selected ICA components
DataTest = ica_images(:, :, top_components); % Use selected components

% Further normalization after PCA
[H, W, Dim] = size(DataTest);
num = H * W;
for i = 1:Dim
    DataTest(:, :, i) = (DataTest(:, :, i) - min(min(DataTest(:, :, i)))) / ...
        (max(max(DataTest(:, :, i))) - min(min(DataTest(:, :, i))));
end
mask =PlaneGT;
%%%% Reshape for anomaly detection
mask_reshape = reshape(mask, 1, num);
anomaly_map = logical(double(mask_reshape) > 0);
normal_map = logical(double(mask_reshape) == 0);
Y = reshape(DataTest, num, Dim)';

X = DataTest;
[n1, n2, n3] = size(X);

%% Test R-TPCA
opts.lambda = 0.2; % Adjust as needed for your dataset
opts.mu = 1e-4;
opts.tol = 1e-8;
opts.rho = 1.1;
opts.max_iter = 100;
opts.DEBUG = 0;
tic;
[L, S, rank] = dictionary_learning_tlrr(X, opts);

%% Test PCA-TLRSR
max_iter = 100;
Debug = 0;
lambda = 0.01; % Adjust as needed for your dataset
[Z, tlrr_E, Z_rank, err_va] = TLRSR(X, L, max_iter, lambda, Debug);

Time_TLRR = toc;

%% Compute Metrics
E = reshape(tlrr_E, num, Dim)';
r_new = sum(abs(E), 1);
r_max = max(r_new(:));

% AUC and ROC
taus = linspace(0, r_max, 5000);
PF_40 = zeros(1, 5000);
PD_40 = zeros(1, 5000);
for index2 = 1:length(taus)
    tau = taus(index2);
    anomaly_map_rx = (r_new > tau);
    PF_40(index2) = sum(anomaly_map_rx & normal_map) / sum(normal_map);
    PD_40(index2) = sum(anomaly_map_rx & anomaly_map) / sum(anomaly_map);
end
area_TLRR = sum((PF_40(1:end-1) - PF_40(2:end)) .* (PD_40(2:end) + PD_40(1:end-1)) / 2);

% Threshold for binary classification (example: mean threshold)
tau_binary = prctile(r_new, 99); %
anomaly_map_pred = (r_new > tau_binary);

% Reshape maps
anomaly_map_pred = reshape(anomaly_map_pred, H, W);

% --- New: Refine anomalies using the precomputed mask ---
load('similarity_mask.mat', 'similarity_mask'); % Load similarity mask

% Label connected components in the anomaly map
[anomaly_labels, num_anomalies] = bwlabel(anomaly_map_pred);

% Create a new anomaly map that retains only valid anomalies
refined_anomaly_map = zeros(size(anomaly_map_pred));

for i = 1:num_anomalies
    current_anomaly = (anomaly_labels == i);
    if any(current_anomaly & similarity_mask, 'all')
        refined_anomaly_map = refined_anomaly_map | current_anomaly;
    else
        fprintf('Anomaly %d removed: No overlap with mask.\n', i);
    
    end
end

% Visualize refined anomaly map

anomaly_map_gt = reshape(anomaly_map, H, W);

% Detect Bounding Boxes for Likely Drones
% Refine anomalies using the precomputed mask
load('similarity_mask.mat', 'similarity_mask'); % Load similarity mask

% Label connected components in the anomaly map
[anomaly_labels, num_anomalies] = bwlabel(anomaly_map_pred);

% Create a new anomaly map that retains only valid anomalies
refined_anomaly_map = zeros(size(anomaly_map_pred));
overlap_threshold = 0.2;
for i = 1:num_anomalies
    current_anomaly = (anomaly_labels == i);
    total_anomalous_pixels = sum(current_anomaly(:));
    overlapping_pixels = sum((current_anomaly & similarity_mask), 'all');
    
    % Check if the overlap meets the 20% threshold
    if overlapping_pixels / total_anomalous_pixels >= overlap_threshold
        refined_anomaly_map = refined_anomaly_map | current_anomaly; % Keep this anomaly
    end
end

% Visualize refined anomaly map
figure;
imshow(refined_anomaly_map, []);
title('Refined Anomaly Map');

% Use refined anomaly map for bounding box detection
se = strel('disk', 5); % Structuring element with a radius of 5 pixels
anomaly_map_dilated = imdilate(refined_anomaly_map, se); % Use refined map here

% Detect Bounding Boxes for Likely Drones
stats = regionprops(anomaly_map_dilated, 'BoundingBox', 'Area');
all_bboxes = reshape(cell2mat({stats.BoundingBox}), 4, []).';

% Merge overlapping bounding boxes
for i = 1:size(all_bboxes, 1)
    for j = i+1:size(all_bboxes, 1)
        % Check if boxes overlap
        bbox1 = all_bboxes(i, :);
        bbox2 = all_bboxes(j, :);
        if rectint(bbox1, bbox2) > 0
            % Merge boxes
            x_min = min(bbox1(1), bbox2(1));
            y_min = min(bbox1(2), bbox2(2));
            x_max = max(bbox1(1) + bbox1(3), bbox2(1) + bbox2(3));
            y_max = max(bbox1(2) + bbox1(4), bbox2(2) + bbox2(4));
            merged_bbox = [x_min, y_min, x_max - x_min, y_max - y_min];
            all_bboxes(i, :) = merged_bbox;
            all_bboxes(j, :) = NaN; % Mark the second box as merged
        end
    end
end
% Remove merged (NaN) entries
all_bboxes = all_bboxes(~isnan(all_bboxes(:, 1)), :);

% Visualize all bounding boxes before filtering
figure;
imshow(refined_anomaly_map, []); % Use refined map here
hold on;
for k = 1:size(all_bboxes, 1)
    rectangle('Position', all_bboxes(k, :), 'EdgeColor', 'b', 'LineWidth', 1); % All bounding boxes
end
hold off;
title('All Bounding Boxes After Merging (Blue: Before Filtering)');

% Filter bounding boxes by size and number of anomalous pixels
min_area = 100; % Minimum area in pixels
max_area = 3200; % Maximum area in pixels
min_anomalous_pixels = 30; % Minimum number of anomalous pixels
filtered_bboxes = [];
for k = 1:size(all_bboxes, 1)
    bbox = all_bboxes(k, :);
    bbox_area = bbox(3) * bbox(4);
    
    % Calculate number of anomalous pixels in the bounding box
    x_start = max(floor(bbox(1)), 1); % Ensure indices are >= 1
    y_start = max(floor(bbox(2)), 1); % Ensure indices are >= 1
    x_end = min(ceil(bbox(1) + bbox(3)), W); % Ensure indices are <= image width
    y_end = min(ceil(bbox(2) + bbox(4)), H); % Ensure indices are <= image height
    num_anomalous_pixels = sum(sum(refined_anomaly_map(y_start:y_end, x_start:x_end))); % Use refined map here
    density_threshold = 0.05;
    if bbox_area >= min_area && bbox_area <= max_area && num_anomalous_pixels >= min_anomalous_pixels
    % Add the bounding box to the filtered list if it meets all conditions
        if bbox_area >= min_area && bbox_area <= max_area && num_anomalous_pixels >= min_anomalous_pixels
            % Sparsity Filtering (Additional Check)
            bbox_region = refined_anomaly_map(y_start:y_end, x_start:x_end);
            
            % Label connected components in the bounding box
            [labeled_region, num_components] = bwlabel(bbox_region);
            
            % Analyze connected components
            component_areas = regionprops(labeled_region, 'Area');
            component_sizes = [component_areas.Area];
            
            % Calculate anomaly density
            
            anomaly_density = num_anomalous_pixels / bbox_area;
            
            % Count large connected components
            
            % Sparsity Criteria: Check Density and Connected Components
            if anomaly_density >= density_threshold
                % Add bounding box to filtered list if it meets sparsity criteria
                filtered_bboxes = [filtered_bboxes; bbox];
            else
                fprintf('Bounding box rejected due to sparsity (Density: %.2f, Large Components: %d).\n', ...
                    anomaly_density);
            end
        end

    else
        % Debugging: Print why the condition failed
        fprintf('Bounding box rejected: [%f, %f, %f, %f]\n', bbox(1), bbox(2), bbox(3), bbox(4));
        
        % Check each condition and print the reason for failure
        if bbox_area < min_area
            fprintf('  Reason: Area too small (%.2f < %.2f)\n', bbox_area, min_area);
        elseif bbox_area > max_area
            fprintf('  Reason: Area too large (%.2f > %.2f)\n', bbox_area, max_area);
        end
        
        
    end
end

% Draw Bounding Boxes with Ground Truth
GT_stats = regionprops(anomaly_map_gt, 'BoundingBox');
ground_truth_bboxes = reshape(cell2mat({GT_stats.BoundingBox}), 4, []).';

figure;
imshow(refined_anomaly_map, []); % Use refined map here
hold on;
for k = 1:size(filtered_bboxes, 1)
    rectangle('Position', filtered_bboxes(k, :), 'EdgeColor', 'r', 'LineWidth', 2); % Predicted
end
for k = 1:size(ground_truth_bboxes, 1)
    rectangle('Position', ground_truth_bboxes(k, :), 'EdgeColor', 'g', 'LineWidth', 2); % Ground Truth
end
hold off;
title('Filtered Anomaly Map with Bounding Boxes (Red: Predicted, Green: Ground Truth)');

% Compute Metrics Based on Bounding Boxes
TP = 0; FP = 0; FN = 0;
IoU_values = [];
matched_predictions = false(size(filtered_bboxes, 1), 1); % Track matched predicted bboxes

for i = 1:size(ground_truth_bboxes, 1)
    gt_bbox = ground_truth_bboxes(i, :);
    best_IoU = 0;
    best_pred_idx = -1;
    
    for j = 1:size(filtered_bboxes, 1)
        if matched_predictions(j)
            continue; % Skip already matched predicted bboxes
        end
        
        pred_bbox = filtered_bboxes(j, :);
        % Compute Intersection over Union (IoU)
        intersection_area = rectint(gt_bbox, pred_bbox);
        union_area = (gt_bbox(3) * gt_bbox(4)) + (pred_bbox(3) * pred_bbox(4)) - intersection_area;
        IoU = intersection_area / union_area;
        
        % Check if bbox is fully contained within ground truth
        is_fully_contained = (pred_bbox(1) >= gt_bbox(1)) && ...
                             (pred_bbox(2) >= gt_bbox(2)) && ...
                             (pred_bbox(1) + pred_bbox(3) <= gt_bbox(1) + gt_bbox(3)) && ...
                             (pred_bbox(2) + pred_bbox(4) <= gt_bbox(2) + gt_bbox(4));
                             
        % Update best match if IoU is higher or bbox is fully contained
        if IoU > best_IoU || (is_fully_contained && IoU < 0.5)
            best_IoU = IoU;
            best_pred_idx = j;
        end
    end
    
    if best_pred_idx ~= -1
        % Match found
        TP = TP + 1;
        IoU_values = [IoU_values; best_IoU];
        matched_predictions(best_pred_idx) = true;
    else
        % No match found for this ground truth bbox
        FN = FN + 1;
    end
end

% Count unmatched predicted bboxes as false positives
FP = sum(~matched_predictions);

% Calculate Metrics
Precision = TP / max((TP + FP), 1); % Avoid division by zero
Recall = TP / max((TP + FN), 1);
F1_score = 2 * (Precision * Recall) / max((Precision + Recall), 1);
Mean_IoU = mean(IoU_values);

% Print Metrics
fprintf('Precision: %.4f\n', Precision);
fprintf('Recall: %.4f\n', Recall);
fprintf('F1 Score: %.4f\n', F1_score);
fprintf('Mean IoU: %.4f\n', Mean_IoU);

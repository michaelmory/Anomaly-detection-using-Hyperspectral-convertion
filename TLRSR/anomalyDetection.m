
function bounding_boxes = anomalyDetection(rec_hs, numb_pca_components, num_ica_components, ICA_max_iterations, similarity_threshold, opts_lambda, opts_max_iter, lambda, overlap_threshold, min_area, max_area, min_anomalous_pixels)
    
    addpath(genpath('./tSVD'));
    addpath(genpath('./proximal_operator'));
    addpath(genpath('./FastICA_25'));
    
    data = rec_hs; % Hyperspectral image from your .mat file
    % --- New: Load the average spectrum ---
    load('average_spectrum.mat', 'average_spectrum'); % Precomputed from multiple images
    
    % --- New: Generate similarity mask using the original data ---
    [rows, cols, ~] = size(data);
    distance_map = zeros(rows, cols);
    
    for i = 1:rows
        for j = 1:cols
            current_spectrum = squeeze(data(i, j, :));
            distance_map(i, j) = norm(current_spectrum - average_spectrum);
        end
    end
    
    % Define a threshold based on the original data
    
    similarity_mask = distance_map <= similarity_threshold;
    % Visualize the similarity mask
    
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
    [~, score] = pca(X); % PCA transformation
    reduced_X = score(:, 1:numb_pca_components); % Retain top PCA components
    
    % Apply ICA on PCA-reduced data
    [icasig, ~, ~] = fastica(reduced_X', 'numOfIC', num_ica_components, 'maxNumIterations', ICA_max_iterations);
    
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
    
    %[n1, n2, n3] = size(X);
    
    %% Test R-TPCA
    opts.lambda = opts_lambda; % Adjust as needed for your dataset
    opts.mu = 1e-4;
    opts.tol = 1e-8;
    opts.rho = 1.1;
    opts.max_iter = opts_max_iter;
    opts.DEBUG = 0;
    tic;
    [L, ~, ~] = dictionary_learning_tlrr(DataTest, opts);
    
    %% Test PCA-TLRSR
    max_iter = 100;
    Debug = 0;
    [~, tlrr_E, ~, ~] = TLRSR(DataTest, L, max_iter, lambda, Debug);
    
    
    
    E = reshape(tlrr_E, num, Dim)';
    r_new = sum(abs(E), 1);
    % Threshold for binary classification (example: mean threshold)
    tau_binary = prctile(r_new, 99); %
    anomaly_map_pred = (r_new > tau_binary);
    
    % Reshape maps
    anomaly_map_pred = reshape(anomaly_map_pred, H, W);
   
    
    % Label connected components in the anomaly map
    [anomaly_labels, num_anomalies] = bwlabel(anomaly_map_pred);
    figure;
    imshow(anomaly_map_pred, []);
    title('Refined Anomaly Map');
    % Create a new anomaly map that retains only valid anomalies
    refined_anomaly_map = zeros(size(anomaly_map_pred));
    for i = 1:num_anomalies
        current_anomaly = (anomaly_labels == i);
        total_anomalous_pixels = sum(current_anomaly(:));
        overlapping_pixels = sum((current_anomaly & similarity_mask), 'all');
        
        % Check if the overlap meets the 20% threshold
        if overlapping_pixels / total_anomalous_pixels >= overlap_threshold
            refined_anomaly_map = refined_anomaly_map | current_anomaly; % Keep this anomaly
        end
    end
    
  
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
   
   
    % Filter bounding boxes by size and number of anomalous pixels
    
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
        if bbox_area >= min_area && bbox_area <= max_area && num_anomalous_pixels >= min_anomalous_pixels
        % Add the bounding box to the filtered list if it meets all conditions
            filtered_bboxes = [filtered_bboxes; bbox]; 
        end
    end
    bounding_boxes = filtered_bboxes;
end

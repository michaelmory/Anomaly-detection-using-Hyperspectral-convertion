% Clear workspace
clear all;
close all;
clc;

% Load an example hyperspectral image (or a regular image for simplicity)
% Replace 'your_image.mat' with your image file or use a standard image
load('1954.mat'); % Hyperspectral image (HxWxDim)
data = rec_hs; % Adjust this variable to your image data

dims = size(data); % Get all dimensions
H = dims(1); % Height
W = dims(2); % Width
Dim = dims(3); % Depth (number of spectral bands or features)
% Reshape the image data into a 2D matrix for ICA
% Rows: Pixels; Columns: Bands/Features

X = reshape(data, [], Dim); % Shape: [H*W x Dim]

% Center the data (subtract mean for each band)
X = bsxfun(@minus, X, mean(X, 1));
[coeff, score] = pca(X);
reduced_X = score(:, 1:10); % Retain top 10 principal components

% Apply ICA on PCA-reduced data
[icasig, ~, ~] = fastica(reduced_X', 'numOfIC', 10);
% Define the number of components to extract (e.g., 10)
num_components = 20;

% Run FastICA
%[icasig, A, B] = fastica(X', 'numOfIC', num_components);

% Validate reshape compatibility
actual_num_components = size(icasig, 1);
if actual_num_components < num_components
    warning(['FastICA converged with fewer components than requested: ', ...
             num2str(actual_num_components), ' out of ', num2str(num_components)]);
    num_components = actual_num_components; % Update num_components
end


% Reshape independent components back to image size
ica_images = reshape(icasig', H, W, num_components);

% Visualize the independent components
for i = 1:num_components
    figure; % Create a new figure for each component
    imagesc(ica_images(:, :, i)); % Display the component as an image
    colormap('jet'); % Use a colormap to represent values
    colorbar; % Add a colorbar for scale
    title(['ICA Component ', num2str(i)]); % Title for the figure
    axis image off; % Keep the aspect ratio and turn off axes
end
 %Step 1: Calculate variance and kurtosis for each component
variance_values = zeros(1, num_components);
kurtosis_values = zeros(1, num_components);

for i = 1:num_components
    component = ica_images(:, :, i);
    % Variance
    variance_values(i) = var(component(:));
    % Kurtosis
    kurtosis_values(i) = kurtosis(component(:));
end

% Select top components based on variance
[~, top_variance_indices] = sort(variance_values, 'descend');
top_variance_indices = top_variance_indices(1:3); % Select top 3 components

% Select top components based on kurtosis
[~, top_kurtosis_indices] = sort(kurtosis_values, 'descend');
top_kurtosis_indices = top_kurtosis_indices(1:3); % Select top 3 components



% Visualization of selected components based on kurtosis
for i = 1:length(top_kurtosis_indices)
    figure;
    imagesc(ica_images(:, :, top_kurtosis_indices(i)));
    colormap('jet');
    colorbar;
    title(['Selected by Kurtosis - Component ', num2str(top_kurtosis_indices(i)), ...
        ', Kurtosis: ', num2str(kurtosis_values(top_kurtosis_indices(i)))]);
    axis image off;
end



% Sliding window size for entropy calculation
window_size = 31; % Adjust based on expected drone dimensions

% Initialize scores for entropy-based selection
entropy_scores = zeros(1, num_components);

for i = 1:num_components
    component = ica_images(:, :, i);
    
    % Normalize component to [0, 1] for entropy calculation
    component = (component - min(component(:))) / (max(component(:)) - min(component(:)));
    
    % Calculate local entropy with sliding window
    local_entropy = entropyfilt(component, true(window_size));
    
    % Calculate the maximum entropy value within the component
    entropy_scores(i) = max(local_entropy(:)); % Focus on the most "informative" region
end

% Select top components based on entropy scores
[~, top_entropy_indices] = sort(entropy_scores, 'descend');
top_components = top_entropy_indices(1:3); % Select top 3 components

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
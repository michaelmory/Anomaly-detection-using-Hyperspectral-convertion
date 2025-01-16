% Specify the files containing the hyperspectral images
image_files = {'test.mat', 'pos_G3P51915.mat', 'pos_G3P15551.mat'}; % Replace with your .mat file names

% Specify the row and column indices of the selected pixels for each image
selected_pixels = {
    [186, 311]; % [row, col] for image1
    [182, 248]; % [row, col] for image2
    [210, 331]  % [row, col] for image3
};

% Initialize a matrix to store the selected spectra
num_images = numel(image_files);
num_bands = []; % Placeholder for number of bands (inferred from the first image)
selected_spectra = [];

% Loop through each image and extract the spectrum for the selected pixel
for i = 1:num_images
    % Load the image
    load(image_files{i});
    
    if ~exist('rec_hs', 'var')
        error("The variable 'rec_hs' is not found in %s.", image_files{i});
    end
    
    % Extract the pixel spectrum
    row = selected_pixels{i}(1);
    col = selected_pixels{i}(2);
    
    if isempty(num_bands)
        num_bands = size(rec_hs, 3);
    end
    
    spectrum = squeeze(rec_hs(row, col, :));
    selected_spectra = [selected_spectra; spectrum']; % Append the spectrum as a row
end
% Plot the spectra from all selected pixels

% Compute the average spectrum across the selected pixels
average_spectrum = mean(selected_spectra, 1);
save('average_spectrum.mat', 'average_spectrum');
% Load the target image to apply the mask
load('1954.mat'); % Replace with your target image file name
if ~exist('rec_hs', 'var')
    error("The variable 'rec_hs' is not found in the target file.");
end
target_hs = rec_hs; % Assume the data is in 'rec_hs'

% Compute the similarity mask for the target image
[rows, cols, bands] = size(target_hs);
distance_map = zeros(rows, cols);

for i = 1:rows
    for j = 1:cols
        % Spectrum of the current pixel in the target image
        current_spectrum = squeeze(target_hs(i, j, :));
        % Euclidean distance to the average spectrum
        distance_map(i, j) = norm(current_spectrum - average_spectrum);
    end
end

% Define a threshold for similarity
threshold = 0.3; % Adjust based on your data

% Create the mask for the target image
mask = distance_map <= threshold;

% Display the mask on the target image
figure;
imshow(mask, []);
title('Generated Mask for Target Image (Multiple Pixels from Multiple Images)');

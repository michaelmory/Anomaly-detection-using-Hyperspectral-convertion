% Directory containing .mat files
source_directory = 'hs';
output_directory = 'hs_205';

% Define target band count and range
desired_band_count = 205;
band_range = [400, 2500];
desired_bands = linspace(band_range(1), band_range(2), desired_band_count);

% Process each file
files = dir(fullfile(source_directory, '*.mat'));
for i = 1:length(files)
    % Load the file
    file_path = fullfile(source_directory, files(i).name);
    file_data = load(file_path);

    % Check if the file contains hyperspectral data
    if isfield(file_data, 'data')
        rad = file_data.data; % Extract hyperspectral data
    else
        fprintf('File %s does not contain "data". Skipping...\n', files(i).name);
        continue;
    end

    % Resample if the band count is inconsistent
    [height, width, original_band_count] = size(rad);
    if original_band_count ~= desired_band_count
        original_bands = linspace(band_range(1), band_range(2), original_band_count);
        reshaped_rad = reshape(rad, [], original_band_count); % Flatten spatial dimensions
        resampled_rad = interp1(original_bands, reshaped_rad', desired_bands, 'linear', 'extrap')'; % Interpolation
        rad_resampled = reshape(resampled_rad, height, width, desired_band_count); % Reshape back to 3D
    else
        rad_resampled = rad; % No resampling needed
    end

    % Save the resampled image
    output_file = fullfile(output_directory, sprintf('resampled_image_%d.mat', i));
    save(output_file, 'rad_resampled', 'desired_bands');
    fprintf('Processed and saved: %s\n', output_file);
end

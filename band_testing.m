% Directory containing your resampled .mat files
source_directory = 'hs_205'; % Resampled files directory
output_directory = 'preprocessed_HS'; % Preprocessed files directory

% Define inferred band range for 205 bands (adjust if needed)
total_bands = 205; % Matches the number of bands after resampling
band_range = [400, 2500]; % Hypothetical range for 205 bands
inferred_bands = linspace(band_range(1), band_range(2), total_bands);

% Process each .mat file
files = dir(fullfile(source_directory, '*.mat'));
for i = 1:length(files)
    % Load the file
    file_path = fullfile(source_directory, files(i).name);
    file_data = load(file_path);

    % Check if the file contains resampled hyperspectral data
    if isfield(file_data, 'rad_resampled')
        rad = file_data.rad_resampled; % Extract the resampled hyperspectral data
    else
        fprintf('File %s does not contain "rad_resampled". Skipping...\n', files(i).name);
        continue;
    end

    % Save the preprocessed data with inferred bands
    output_file = fullfile(output_directory, sprintf('preprocessed_image_%d.mat', i));
    save(output_file, 'rad', 'inferred_bands');
    fprintf('Processed and saved: %s\n', output_file);
end

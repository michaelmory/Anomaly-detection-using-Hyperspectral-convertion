clear all;
close all;
clc;

% Load your hyperspectral image data (same as in the original script)
load('1954.mat'); % Replace with the name of your .mat file

% Extract hyperspectral image
data = rec_hs; % Hyperspectral image from your .mat file
[H, W, Dim] = size(data);

% Analyze Spectral Profiles for a Sample Pixel (e.g., Drone and Cloud)
% Define sample pixel indices for the drone and a cloud region
row_drone = 170; col_drone = 376; % Example indices for the drone pixel
row_cloud = 252; col_cloud = 505; % Example indices for a cloud pixel

% Extract and plot spectral profiles
spectral_drone = squeeze(data(row_drone, col_drone, :));
spectral_cloud = squeeze(data(row_cloud, col_cloud, :));

figure;
plot(1:Dim, spectral_drone, '-r', 'LineWidth', 1.5);
hold on;
plot(1:Dim, spectral_cloud, '-b', 'LineWidth', 1.5);
title('Spectral Profiles of Drone and Cloud');
xlabel('Band Index');
ylabel('Reflectance');
legend('Drone', 'Cloud');
grid on;
hold off;

% Statistical Analysis Across Bands
% Compute mean and standard deviation for each band
band_means = squeeze(mean(mean(data, 1), 2));
band_stds = squeeze(std(std(data, 0, 1), 0, 2));

figure;
plot(1:Dim, band_means, '-o', 'LineWidth', 1.5);
hold on;
plot(1:Dim, band_stds, '-x', 'LineWidth', 1.5);
title('Mean and Standard Deviation of Spectral Bands');
xlabel('Band Index');
ylabel('Value');
legend('Mean', 'Std Dev');
grid on;
hold off;
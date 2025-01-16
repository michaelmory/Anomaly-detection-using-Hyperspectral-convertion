clear all;
close all;
clc;

% Load your hyperspectral image data (same as in the original script)
load('1954.mat'); % Replace with the name of your .mat file

% Extract hyperspectral image
data = rec_hs; % Hyperspectral image from your .mat file

% Display the mean reflectance image
figure;
imagesc(mean(data, 3)); % Compute and display the mean reflectance across bands
title('Mean Reflectance of Hyperspectral Image');
xlabel('X (Columns)');
ylabel('Y (Rows)');
colormap('jet'); % Use a visually distinct colormap
grid on;
colorbar; % Add a colorbar for reference

% Add dependencies
addpath('ompbox10');
addpath('ksvdbox13');
assert(exist('omp','file') == 2, 'OMP-Box not found, cannot continue.');
assert(exist('ksvd','file') == 2, 'KSVD-Box not found, cannot continue.');

% Load precomputed dictionaries
disp('Loading dictionaries and image');
% load('sample_dict.mat'); % Provides 'Dic_HS' and 'Dic_Cam'

% Load ground truth HS image
load('sample_hs_im.mat'); % Provides 'rad' and 'bands'
% rad = (rad ./ max(rad(:))) * 4095; % "stretch" HS image to full luminance range

% Load CIE 1964 color matching function (target camera);
load('cie_1964_400_700.mat'); % Provides cie_1964

% Load a user-selected camera image
[filename, pathname] = uigetfile({'*.jpg;*.png;*.bmp', 'Image Files (*.jpg, *.png, *.bmp)'}, 'Select a Camera Image');
if isequal(filename, 0)
    error('No image selected');
else
    user_image = imread(fullfile(pathname, filename));
end

% Resize to match HS dimensions if necessary
target_size = size(rad(:,:,1));
user_image_resized = imresize(user_image, target_size(1:2));

% Use the resized RGB image as input for the hyperspectral reconstruction
im_cam = double(user_image_resized);

% Apply camera response function to HS data (if simulating camera is needed)
% Uncomment if you want to simulate a camera response from the ground truth data
% disp('Preparing simulated camera image');
% im_cam = shredProjectImage(rad, bands, cie_1964);

% Reconstruct HS information from the camera image
fprintf('Reconstructing HS image from camera image...');
sparsity_target = 28; % Adjust sparsity target for reconstruction
rec_hs = shredReconstructImage(im_cam, Dic_Cam, Dic_HS, sparsity_target);
fprintf('Done\n');

% Compute error metrics: average RMSE and average RRMSE
RMSE = sqrt(mean((rec_hs(:) - rad(:)).^2));
RMSE = (RMSE / max(rad(:))) * 255; % RMSE on 0-255 scale
RRMSE = shredRRMSE(rec_hs, rad);
disp(['RMSE: ' num2str(RMSE) '   RRMSE: ' num2str(RRMSE)]);

% Partial visualization of results:
figure(1);
visualized_bands=[7,16,23];
for i=1:3

    subplot(3,3,3+i);
    imagesc(rec_hs(:,:,visualized_bands(i)));
    title([num2str(bands(visualized_bands(i))) 'nm Reconstructed']);
    colormap bone; axis image; axis off;
    
    subplot(3,3,6+i);
    imagesc( ((rad(:,:,visualized_bands(i))-rec_hs(:,:,visualized_bands(i)))./max(rad(:)))*255, [-20 20]  );
    title('Error map');
    axis image; axis off;
end
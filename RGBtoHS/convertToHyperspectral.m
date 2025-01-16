function rec_hs = convertToHyperspectral(image_path,dict)
    % RECONSTRUCTHYPERSPECTRALIMAGE Reconstructs a hyperspectral image from an RGB image.
    %   rec_hs = reconstructHyperspectralImage(dict, image_path, sparsity_target)
    %   Inputs:
    %       - dict: A structure containing Dic_Cam, Dic_HS, and cie_1964.
    %       - image_path: Path to the RGB image file.
    %       - sparsity_target: Sparsity target for the reconstruction.
    %   Output:
    %       - rec_hs: Reconstructed hyperspectral image.
    sparsity_target = 28;
    % Validate dictionary fields
    data = load(dict, 'Dic_HS', 'Dic_Cam', 'cie_1964');

    % Create a structure to pass to the function
    Dic_HS = data.Dic_HS;
    Dic_Cam = data.Dic_Cam;

    % Read and preprocess the RGB image
    im_cam = imread(image_path);
    im_cam = im2double(im_cam); % Convert the image to double format [0, 1]

    rec_hs = shredReconstructImage(im_cam,Dic_Cam, Dic_HS , sparsity_target);
end

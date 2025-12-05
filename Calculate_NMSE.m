%% Calculate NMSE from Saved Model
clear; clc;
addpath("data/", "channels/", "functions/", "classes/");

% 1. Load Model
model_file = 'trained_LAMP_model.mat';
if ~isfile(model_file)
    error('Model file %s not found. Run DU_Training.m first.', model_file);
end
load(model_file, 'lamp_layers', 'norm_factor');
fprintf('Loaded model: %s\n', model_file);
fprintf('Loaded norm_factor: %.2e\n', norm_factor);

% 2. System Parameters (Must match training)
fc = 300e9;
c = 3e8;
num_sc = 32;
Nr = 256;
lambda_c = c/fc;
d = lambda_c/2;

% 3. Load Test Data
fprintf('Loading test data...\n');
SNR_collection = [0, 5, 10, 15, 20];
Mr_collection = [16, 32, 64, 128];
channel_model = 'cluster';

Y_test_list = {};
count = 0;
for Mr = Mr_collection
    for SNR = SNR_collection
        fname = sprintf('data/data_%dBeams_%ddB-SNR_%s_%dscs.mat', Mr, SNR, channel_model, num_sc);
        if isfile(fname)
            count = count + 1;
            loaded = load(fname, 'H_list');
            H_batch = loaded.H_list; % (N, num_sc, Nr)
           
            H_perm = permute(H_batch, [2, 3, 1]);
            % Cast to single precision to save memory
            H_final = cat(3, reshape(real(H_perm), num_sc, Nr, 1, []), ...
                             reshape(imag(H_perm), num_sc, Nr, 1, []));
            Y_test_list{end+1} = single(H_final); %#ok<*SAGROW>
        end
    end
end

if isempty(Y_test_list)
    error('No data files found.');
end

H_target_all = cat(4, Y_test_list{:});

% Apply normalization (CRITICAL: must match training!)
if norm_factor > 0
    H_target_all = H_target_all / norm_factor;
    fprintf('Applied normalization (factor: %.2e)\n', norm_factor);
else
    warning('norm_factor is zero or invalid. Skipping normalization.');
end

Y_eval = dlarray(H_target_all, 'SSCB');
[n_sc, nr, chan, n_samples] = size(Y_eval);
fprintf('Data Loaded. Total Samples: %d\n', n_samples);

% 4. Forward Pass (with Batch Processing to avoid OOM)
fprintf('Running inference...\n');
num_layers = length(lamp_layers);
G = length(lamp_layers{1}.Thetas);

% Batch processing parameters
batch_size = 100; % Process 100 samples at a time
num_batches = ceil(n_samples / batch_size);

% Preallocate output for final h
h_all = zeros(n_sc, G, 2, n_samples, 'single');

for batch_idx = 1:num_batches
    % Calculate batch indices
    start_idx = (batch_idx - 1) * batch_size + 1;
    end_idx = min(batch_idx * batch_size, n_samples);
    batch_samples = end_idx - start_idx + 1;
    
    fprintf('Processing batch %d/%d (samples %d-%d)...\n', batch_idx, num_batches, start_idx, end_idx);
    
    % Extract batch data
    Y_batch = Y_eval(:, :, :, start_idx:end_idx);
    
    % Initialize h and v for this batch
    h = dlarray(zeros(n_sc, G, 2, batch_samples, 'single'), 'SSCB');
    v = Y_batch;
    
    % Forward pass through all layers
    for k = 1:num_layers
        [h, v] = lamp_layers{k}.predict(h, v, Y_batch);
    end
    
    % Store results
    h_all(:, :, :, start_idx:end_idx) = extractdata(h);
end

% Convert final h to dlarray for reconstruction
h = dlarray(h_all, 'SSCB');
fprintf('Inference complete.\n');

% 5. Reconstruct Output (Y_pred = A * h)
fprintf('Reconstructing signal...\n');
final_layer = lamp_layers{end};

% --- FIX: Force parameters to be real ---
% This prevents NaNs if slight imaginary noise exists in the saved parameters
final_thetas = real(final_layer.Thetas);
final_ranges = real(final_layer.Ranges);

A_complex = differentiable_manifold(final_thetas, final_ranges, ...
                                    Nr, d, fc, num_sc);

% Reshape A: (Nr*num_sc, G) -> (Nr, G, num_sc)
A_reshaped = reshape(A_complex, [Nr, num_sc, G]);
A_perm = permute(A_reshaped, [1, 3, 2]);
Ar = real(A_perm);
Ai = imag(A_perm);

% Reshape h: (num_sc, G, 2, Batch) -> (G, Batch, num_sc)
h_raw = stripdims(h);
hr_3d = reshape(h_raw(:,:,1,:), num_sc, G, n_samples);
hi_3d = reshape(h_raw(:,:,2,:), num_sc, G, n_samples);
hr = permute(hr_3d, [2, 3, 1]);
hi = permute(hi_3d, [2, 3, 1]);

% Multiply Y = A * h
yr = pagemtimes(Ar, hr) - pagemtimes(Ai, hi);
yi = pagemtimes(Ar, hi) + pagemtimes(Ai, hr);

% Reshape back to target format
yr = permute(yr, [3, 1, 2]);
yi = permute(yi, [3, 1, 2]);
Y_pred = cat(3, reshape(yr, num_sc, Nr, 1, n_samples), ...
                reshape(yi, num_sc, Nr, 1, n_samples));

% 6. Calculate NMSE
% Denormalize predictions and ground truth for correct NMSE calculation
Y_pred_denorm = Y_pred * norm_factor;
Y_eval_denorm = Y_eval * norm_factor;

% NMSE = ||Y_true - Y_pred||^2 / ||Y_true||^2
diff = Y_eval_denorm - Y_pred_denorm;
mse_val = sum(diff.^2, 'all');
power_val = sum(Y_eval_denorm.^2, 'all');

% Check for 0 power to avoid divide-by-zero NaN
if power_val == 0
    warning('Signal power is zero. Cannot calculate NMSE.');
    nmse_db = NaN;
else
    nmse_linear = mse_val / power_val;
    nmse_linear_val = double(gather(extractdata(nmse_linear)));
    nmse_db = 10 * log10(nmse_linear_val);
end

nmse_result = nmse_db;
fprintf('\n==============================\n');
fprintf('Final NMSE: %.4f dB\n', nmse_result);
fprintf('==============================\n');
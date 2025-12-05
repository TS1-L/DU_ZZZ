%% Train Deep Unfolding (Parametric LAMP) - GeForce 940MX (Full Run)
% Features: 
% - Batch Size 12 (Safe for 940MX VRAM)
% - dB Loss Plotting (Real-time performance view)
% - Data Normalization included

clear; clc; close all;
addpath("data/", "channels/", "functions/", "classes/");

% --- 0. GPU Setup ---
if canUseGPU
    g = gpuDevice(1);
    reset(g); % Clear VRAM
    fprintf('GPU Detected: %s (VRAM: %.2f GB)\n', g.Name, g.AvailableMemory/1e9);
    use_gpu = true;
else
    warning('No supported GPU found. Falling back to CPU.');
    use_gpu = false;
end

% --- 1. Configuration & Hyperparameters ---
network_lr = 6e-4;  
dict_lr    = 6e-6;  
grad_clip  = 2.0;

% 940MX CONSTRAINT: Batch Size 12
batch_size = 12;    

test = false; % FULL RUN (Set to false for the 14-hour run)

if test
    epochs     = 2;     
    num_layers = 4;     
else
    epochs     = 10;    
    num_layers = 8;     
end

% --- 2. Load Data & NORMALIZE ---
fprintf('Loading Data...\n');
SNR_collection = [0, 5, 10, 15, 20];
Mr_collection = [16, 32, 64, 128];
num_sc = 32; Nr = 256;
channel_model = 'cluster';

Y_train_list = {}; 
for Mr = Mr_collection
    for SNR = SNR_collection
        if test
            fname = sprintf('data/data_%dBeams_%ddB-SNR_%s_%dscs_quicktest.mat', Mr, SNR, channel_model, num_sc);
        else
            fname = sprintf('data/data_%dBeams_%ddB-SNR_%s_%dscs.mat', Mr, SNR, channel_model, num_sc);
        end
        
        if isfile(fname)
            loaded = load(fname, 'H_list');
            H_perm = permute(loaded.H_list, [2, 3, 1]); 
            
            % Convert to SINGLE immediately
            H_final = cat(3, reshape(real(H_perm), num_sc, Nr, 1, []), ...
                             reshape(imag(H_perm), num_sc, Nr, 1, []));
            Y_train_list{end+1} = single(H_final); 
        end
    end
end

if isempty(Y_train_list)
    error('No data found.');
end

H_target_all = cat(4, Y_train_list{:});
H_target = dlarray(H_target_all, 'SSCB');

% --- CRITICAL: NORMALIZE ---
norm_factor = max(abs(extractdata(H_target)), [], 'all');
fprintf('Data Max Amplitude: %.2e. Normalizing to [-1, 1]...\n', norm_factor);

if norm_factor > 0
    H_target = H_target / norm_factor;
else
    norm_factor = 1.0; 
end
fprintf('Data Loaded & Normalized. Shape: %s\n', mat2str(size(H_target)));


% --- 3. Initialize Physics ---
fc = 300e9; c = 3e8; lambda_c = c/fc; d = lambda_c/2; 
s = 2; G_angle = s * Nr; beta = 1.2; rho_min = 3;
[grid_params, ~] = get_dictionary_parameters(Nr, d, lambda_c, G_angle, fc, beta, rho_min);

% --- 4. Build Network ---
lamp_layers = cell(num_layers, 1);
learnables = cell(num_layers, 1);

if use_gpu; target_str = "GPU"; else; target_str = "CPU"; end
fprintf('Initializing Layers (%s)...\n', target_str);

for k = 1:num_layers
    layer = ParametricLAMPLayer(grid_params, Nr, d, fc, num_sc, 64, "LAMP_"+k);
    
    if use_gpu
        layer.Thetas    = gpuArray(single(layer.Thetas));
        layer.Ranges    = gpuArray(single(layer.Ranges));
        layer.StepSize  = gpuArray(single(layer.StepSize));
        layer.Threshold = gpuArray(single(layer.Threshold));
    else
        layer.Thetas    = single(layer.Thetas);
        layer.Ranges    = single(layer.Ranges);
        layer.StepSize  = single(layer.StepSize);
        layer.Threshold = single(layer.Threshold);
    end
    
    lamp_layers{k} = layer;
    learnables{k}.Thetas    = layer.Thetas;
    learnables{k}.Ranges    = layer.Ranges;
    learnables{k}.StepSize  = layer.StepSize;
    learnables{k}.Threshold = layer.Threshold;
end

% --- 5. Training Loop ---
vel_layers = cell(num_layers, 1); 
lossPlot = animatedline('Color', 'b', 'LineWidth', 1.5); % Blue line

% dB PLOT SETUP:
% No need for 'log' scale on Y-axis because dB is ALREADY logarithmic.
% We use a linear scale to view dB values.
xlabel('Iteration'); 
ylabel('MSE Loss (dB)'); 
title('Training Progress (dB)');
grid on;

iteration = 0;
num_samples = size(H_target, 4);
num_batches = floor(num_samples / batch_size);

fprintf('Starting Training (Batch Size: %d)...\n', batch_size);
total_train_start = tic; 

for epoch = 1:epochs
    idx = randperm(num_samples);
    H_target = H_target(:,:,:,idx);
    
    for b = 1:num_batches
        iteration = iteration + 1;
        idx_batch = (b-1)*batch_size + 1 : b*batch_size;
        
        Y_batch_cpu = H_target(:,:,:,idx_batch);
        if use_gpu; Y_batch = gpuArray(Y_batch_cpu); else; Y_batch = Y_batch_cpu; end
        
        % A. Gradients
        [loss, grads] = dlfeval(@model_loss, learnables, lamp_layers, Y_batch, Nr, d, fc, num_sc);
        
        if isnan(extractdata(loss))
            error('DIVERGENCE: Loss is NaN at Iteration %d.', iteration);
        end
        
        % B. Updates
        for k = 1:num_layers
            layer_grads = grads{k};
            if isempty(vel_layers{k})
                init_val = cast(0, 'like', learnables{k}.Thetas);
                vel_layers{k} = struct('Thetas',init_val, 'Ranges',init_val, 'StepSize',init_val, 'Threshold',init_val);
            end
            
            g_thetas = clip_gradient(real(layer_grads.Thetas), grad_clip);
            g_ranges = clip_gradient(real(layer_grads.Ranges), grad_clip);
            g_step   = clip_gradient(layer_grads.StepSize, grad_clip);
            g_thresh = clip_gradient(layer_grads.Threshold, grad_clip);
            
            % Physics Updates
            vel_layers{k}.Thetas = 0.9 * vel_layers{k}.Thetas - dict_lr * g_thetas;
            learnables{k}.Thetas = learnables{k}.Thetas + vel_layers{k}.Thetas;
            
            vel_layers{k}.Ranges = 0.9 * vel_layers{k}.Ranges - dict_lr * g_ranges;
            learnables{k}.Ranges = max(learnables{k}.Ranges + vel_layers{k}.Ranges, 1.0); 
            
            % Gain Updates
            vel_layers{k}.StepSize = 0.9 * vel_layers{k}.StepSize - network_lr * g_step;
            learnables{k}.StepSize = learnables{k}.StepSize + vel_layers{k}.StepSize;
            
            vel_layers{k}.Threshold = 0.9 * vel_layers{k}.Threshold - network_lr * g_thresh;
            learnables{k}.Threshold = max(learnables{k}.Threshold + vel_layers{k}.Threshold, 1e-6);
        end
        
        % GRAPH UPDATE (Convert MSE to dB)
        if mod(iteration, 10) == 0
            curr_loss = double(gather(extractdata(loss)));
            loss_db = 10 * log10(curr_loss); % Convert to dB
            
            addpoints(lossPlot, iteration, loss_db);
            drawnow limitrate;
        end
    end
    fprintf('Epoch %d/%d | Loss: %.2f dB\n', epoch, epochs, loss_db);
end

train_duration = toc(total_train_start); 
D = duration(0,0,train_duration,'Format','hh:mm:ss');
fprintf('Training Complete. Total Time: %s\n', string(D));

% --- 6. Save ---
outfile = 'trained_LAMP_model.mat';

lamp_layers_cpu = lamp_layers;
for k=1:num_layers
    lamp_layers_cpu{k}.Thetas = gather(lamp_layers{k}.Thetas);
    lamp_layers_cpu{k}.Ranges = gather(lamp_layers{k}.Ranges);
    lamp_layers_cpu{k}.StepSize = gather(lamp_layers{k}.StepSize);
    lamp_layers_cpu{k}.Threshold = gather(lamp_layers{k}.Threshold);
end
lamp_layers = lamp_layers_cpu; 

save(outfile, 'lamp_layers', 'grid_params', 'train_duration', 'norm_factor');
fprintf('Saved model to %s\n', outfile);

% --- 7. Verify ---
fprintf('Running NMSE Verification...\n');
run('Calculate_NMSE.m');


%% --- Helpers ---
function g_out = clip_gradient(g_in, threshold)
    g_out = max(min(g_in, threshold), -threshold);
end

function [loss, gradients] = model_loss(learnables, layers, Y_true, Nr, d, fc, num_sc)
    Y_raw = stripdims(Y_true);
    batch_size = size(Y_raw, 4);
    G = length(learnables{1}.Thetas);
    
    h = dlarray(zeros(num_sc, G, 2, batch_size, 'like', Y_true), 'SSCB');
    v = Y_true; 
    
    for k = 1:length(layers)
        layers{k}.Thetas = learnables{k}.Thetas;
        layers{k}.Ranges = learnables{k}.Ranges;
        layers{k}.StepSize = learnables{k}.StepSize;
        layers{k}.Threshold = learnables{k}.Threshold;
        [h, v] = layers{k}.predict(h, v, Y_true);
    end
    
    final = learnables{end};
    A_complex = differentiable_manifold(final.Thetas, final.Ranges, Nr, d, fc, num_sc);
    
    A_reshaped = reshape(A_complex, [Nr, num_sc, G]);
    A_perm = permute(A_reshaped, [1, 3, 2]); 
    Ar = real(A_perm); Ai = imag(A_perm);
    
    h_raw = stripdims(h); 
    hr = permute(reshape(h_raw(:,:,1,:), num_sc, G, batch_size), [2, 3, 1]);
    hi = permute(reshape(h_raw(:,:,2,:), num_sc, G, batch_size), [2, 3, 1]);
    
    yr = pagemtimes(Ar, hr) - pagemtimes(Ai, hi);
    yi = pagemtimes(Ar, hi) + pagemtimes(Ai, hr);
    
    Y_pred = cat(3, reshape(permute(yr,[3,1,2]), num_sc, Nr, 1, batch_size), ...
                    reshape(permute(yi,[3,1,2]), num_sc, Nr, 1, batch_size));
    
    loss = mean((Y_pred - Y_raw).^2, 'all');
    gradients = dlgradient(loss, learnables);
end
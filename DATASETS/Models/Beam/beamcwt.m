%% Load data
beam = readmatrix('beam-1_2-3even.csv')
fs = 364.45 ; % sampling frequency
% Add noise if desired
noise_amp = 0.5 ; 
dm = beam + noise_amp*randn(1, length(beam)) ;


% Wavelet computation
cwt(beam, 'bump', fs) ;
[wt, fresp] = cwt(beam, 'bump', fs) ;
wt_amp = abs(wt) ;
csvwrite('cwt-beam-1_2-3.csv',wt_amp);
csvwrite('freq-beam-1_2-3.csv',fresp)

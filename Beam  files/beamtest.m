%% Load data
beam = readmatrix('beam.csv')
fs = 700 ; % sampling frequency
% Add noise if desired
noise_amp = 0.5 ; 
dm = beam + noise_amp*randn(1, length(beam)) ;


% Wavelet computation
%cwt(dm, 'amor', fs) ;
[wt, fresp] = cwt(dm, 'amor', fs) ;
wt_amp = abs(wt) ;
csvwrite('beamwtnoise.csv',wt_amp);
csvwrite('freqnoise.csv',fresp)

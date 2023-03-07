%% Load data
dof = readmatrix('1DOF.csv')
fs = 60 ; % sampling frequency
% Add noise if desired
noise_amp = 0.5 ; 
dm = dof + noise_amp*randn(1, length(dof)) ;


% Wavelet computation
%cwt(dof, 'bump', fs) ;
[wt, fresp] = cwt(dof, 'bump', fs) ;
wt_amp = abs(wt) ;
csvwrite('DOF1cwt.csv',wt_amp);
csvwrite('freq1DOF.csv',fresp)

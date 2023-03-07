A=open('SineSweep_20170126_101431_PhysUnit_ms2.mat');
data=getfield(A,"data");

f0=getfield(data,"f0");
fend=getfield(data,"fend");

Acc=getfield(data,"Acc");


t0=getfield(data,"t0");
tend=getfield(data,"tend");
fs=length(Acc)/(tend-t0);

for i = 1:11
    r=Acc(i,:);
    %cwt(r, 'bump', fs) ;
    [wt, fresp] = cwt(r, 'bump', fs) ;
    wt_amp = abs(wt) ;
    title = 'exp-'+string(i)+'.csv'
    csvwrite('cwt-'+title,wt_amp);
    csvwrite('freq-'+title,fresp)
end

import numpy as np
# Returns the scaled scalogram.
def scale(scalogram, frequencies, f0, fend):
    import numpy as np
    from scipy.interpolate import interp1d
    Npoint=len(scalogram[0])
    exc_finst = np.linspace(f0, fend, Npoint) # instantaneous excitation frequency

    fnorm = np.linspace(0, 6, 120) # Vector of frequency ratios
    wt_amp_scaled = np.zeros((len(fnorm),Npoint))
    for i in range(Npoint):
      fscaled = (frequencies/exc_finst[i]).reshape(len(scalogram))
      scaling = interp1d(fscaled, scalogram[:,i], kind='linear', 
                         bounds_error=False, fill_value=np.percentile(scalogram[:,i], 25))
      wt_amp_scaled[:,i]=scaling(fnorm)

    wt_amp_scaled2 = np.zeros(np.shape(wt_amp_scaled))
    for i in range(Npoint):
      norm=np.sum(wt_amp_scaled[17:28,i])/5
      wt_amp_scaled2[:,i] = wt_amp_scaled[:,i] / norm
    return wt_amp_scaled2

# Returns the relative amplitude for the i-th harmonic (relative because scaled by the amplitude of the fundamental).
def norm(scalogram, i, scaled=True, frequencies=None, f0=None, fend=None):
  import numpy as np
  if not scaled:
    scaledwt = scale(scalogram, frequencies, f0, fend)
  else:
    scaledwt=scalogram
  l=[]
  Npoint=len(scaledwt[0])
  for j in range(Npoint):
    norm1=np.sum(scaledwt[15:26,j])
    normi=np.sum(scaledwt[20*i-5:20*i+6,j])
    l.append(normi/norm1)
  l=np.array(l)
  return l

# Returns a dictionary, the keys correspond to the interaction and the values are the interaction score for each interaction. Without the use of ML
def interaction2(scalogram, scaled=True, threshold=2, frequencies=None, f0=None, fend=None):
  if not scaled:
    scalogram = scale(scalogram, frequencies, f0, fend)
  n=np.shape(scalogram)[1]
  score={}
  for i in range(2,6):
    l = norm(scalogram, i, scaled=True, frequencies=None, f0=None, fend=None)>threshold
    score[i]=np.sum(l)/n
  return score
import numpy as np
import scipy as sc

def clean_ground_check(y,fs,N,min_int,max_int,T):

    pos_peaks = get_pos_ground_check(y,fs,N,min_int,max_int,T)

    new_y = delete_ground_check(y,pos_peaks)

    return new_y

def get_pos_ground_check(y,fs,N,min_int,max_int,T):
    '''
    y: eeg recording
    fs: feequency sampling step
    N: number of points for smoothing
    min_int: minimal distance in seconds between two peaks of a ground check
    max_int: maximal distance in seconds between two peaks of a ground check
    T: threshold above which a peak might be from a ground check
    '''

    # smooth power of signal
    y2=np.convolve(y**2,np.ones(N)/N,mode='same')

    # detect peaks spaced by given interval corresponding to ground check one
    distance=int(min_int*fs)
    pos_peaks=sc.signal.find_peaks(y2, height=T, threshold=None, distance=distance, prominence=None, width=None, wlen=None, rel_height=0.5, plateau_size=None)[0]

    # get distance between two detected peaks
    distance_list=(pos_peaks[1:]-pos_peaks[:-1])/fs  

    # keep only peaks where the consecutive distance lower than 8.5 s (the minimal distance between two peaks is already selected by min_int)
    indices=np.where(distance_list <= max_int)[0]

    if len(indices) == 0:
        
        return pos_peaks
    
    else:
        index=[]
        for i in indices:
            index.append(i)
            index.append(i+1)
        indices=np.array(index)
        print(indices)
        print(type(indices))
        pos_peaks=pos_peaks[indices]
        # substract pos before the first peak
        pos_peaks[::2]=pos_peaks[::2]-64
        # add pos after the second peak
        pos_peaks[1::2]=pos_peaks[1::2]+64

        return pos_peaks

def delete_ground_check(y,pos_peaks):

    if len(pos_peaks)==0:
        return y
    else:
        new_y=y[:pos_peaks[0]]
        for i in range(1,len(pos_peaks)-1,2):
            new_y=np.concatenate((new_y,y[pos_peaks[i]:pos_peaks[i+1]]))
        new_y=np.concatenate((new_y,y[pos_peaks[-1]:]))

        return new_y
    
def get_windows(y,t,Ws,step,pos_peaks):

    windows=[]
    t_list=[]
    c=0 # initialize index of the pos_peaks
    for k in range(0,len(y)-Ws,step):
        i, j = k, k+Ws
        if i <= pos_peaks[c] < j:

            if pos_peaks[c+1] <= j: # both start and end within the same window
                win=np.concatenate((y[i:pos_peaks[c]],y[pos_peaks[c+1]:j]))
                windows.append(win)
                t_part=np.concatenate((t[i:pos_peaks[c]],t[pos_peaks[c+1]:j]))
                t_list.append(t_part)
            elif pos_peaks[c+1] > j: # start within the window but end in the next one
                win=y[i:pos_peaks[c]]
                windows.append(win)
                t_part=t[i:pos_peaks[c]]
                t_list.append(t_part)
            
        else:
            windows.append(y[i:j])
            t_list.append(t[i:j])

    return t_list, windows


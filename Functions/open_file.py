import numpy as np
import scipy as sc
from Functions.detect_ground_check import clean_ground_check
from Functions.detect_artifacts import *
from Functions.WaveletQuantileNormalization import *
import json

def open_clean_records(name, fs, ground_check=True, artefacts=True):
    '''
    File to open a recording and clean it form ground check and artefacts
    Input: 
    name: name of the recording
    ground_check: True if correction of ground check is wanted, False if not
    artefacts: True if artefacts correction is wanted, False if not
    Output:
    y: the recording free of artefacts and ground checks
    t: the time points associated to the recording
    '''
    #--- load data
    if 'mat' in name:
        data = sc.io.loadmat(name)
        y = data['record'][:,0]

    elif 'log' in name:
        file = open(name,'r')
        lines = file.readlines()
        N = len(lines)
        y = []
        for i in range(N):
            y.append(float(lines[i]))
        y = np.array(y)
    
    elif 'npy' in name:
        y = np.load(name)

    elif 'json' in name:
        with open(name, 'r') as file:
            data = json.load(file)            
            y = np.array(data['MNDRY_EEG_ELEC_POTL_BIS_TEMPR']['waveform'])    

    else:
        print('error when opening the file')

    #--- correct ground check
    if ground_check:
        y = clean_ground_check(y, fs, 16, 7.5, 8.5, 50000)

    #--- adjust EEG
    m_y = np.mean(y)
    if np.abs(m_y) > 10 :
        y = y - m_y

    #--- correct artefacts
    if artefacts:
        list_detection = [2*fs, 1*fs, [0.002,0.02], "sym4", 4, "periodization"] 
        list_WQN = ["sym4", "periodization", 30, 1]
 
        #detection of the artifacts
        index_mask = find_artifacts(y,*list_detection)      

        # if it has some artifacts  
        if len(index_mask) != 0:                                              
        #--- clean the artifacts using WQN algorithm
            y = WQN_3(y,index_mask,*list_WQN)     

    #--- get time points
    N_y = len(y)
    t = np.linspace(0, N_y/fs, N_y)

    return y, t   
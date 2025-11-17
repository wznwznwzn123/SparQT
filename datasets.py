import numpy as np
import h5py
import pickle
import torch


def NOAA():
    
    f = h5py.File('Data/NOAA/sst_weekly.mat','r') 
    sst = np.nan_to_num( np.array(f['sst']) )
    
    num_frames = 1914

    sea = np.zeros((num_frames,180,360,1))
    for t in range(num_frames):
        sea[t,:,:,0] = sst[t,:].reshape(180,360,order='F')
    sea /= sea.max()
    return sea



def pipe():
    
    with open("Data/Turbulent/ch_2Dxysec.pickle", 'rb') as f:
        pipe = pickle.load(f)
        pipe /= np.abs(pipe).max()
    return pipe


def cylinder():
    
    with open('Data/Cylinder/train.pkl', 'rb') as f:
        cyl = pickle.load(f)
        print("length of cyl_train:", len(cyl))
    with open('Data/Cylinder/test.pkl', 'rb') as f:
        cyl_test = pickle.load(f)
        print("length of cyl_test:", len(cyl_test))
    cyl = np.concatenate((cyl, cyl_test), axis=0)
    with open('Data/Cylinder/valid.pkl', 'rb') as f:
        cyl_val = pickle.load(f)
        print("length of cyl_val:", len(cyl_val))
    cyl = np.concatenate((cyl, cyl_val), axis=0)
    cyl = list(cyl)
    print("shape of the first element's data:", cyl[0]['data'].shape)
    for i in range(len(cyl)):
        cyl[i]['data'] = torch.as_tensor(cyl[i]['data']).unsqueeze(-1)
        cyl[i]['data'] = cyl[i]['data'].squeeze(0)
    return cyl
    
def shallow():
    with open('Data/Shallow/train.pkl', 'rb') as f:
        shal = pickle.load(f)
        print("length of shal_train:", len(shal))
    with open('Data/Shallow/test.pkl', 'rb') as f:
        shal_test = pickle.load(f)
        print("length of shal_test:", len(shal_test))
    shal = np.concatenate((shal, shal_test), axis=0)
    shal = list(shal)
    for i in range(len(shal)):
        shal[i]['data'] = torch.as_tensor(shal[i]['data'])
        shal[i]['data'] = shal[i]['data'].squeeze(0)
    return shal
    
def plume():
    with h5py.File('Data/Plume/concentration.h5', "r") as f:
        plume_3D = f['cs']
        plume_3D = np.array(plume_3D)
        plume_3D /= plume_3D.max()
        print("shape of plume_3D:", plume_3D.shape)
    return plume_3D


def porous():
    with h5py.File('Data/Pore/rho_1.h5', "r") as f:
        pore = f['rho'][:]
    return pore
    
def isotropic3D():
    with h5py.File('Isotropic/scalarHIT_fields100.h5', "r") as f:
        return np.array(f['fields'])

def Fire_3D():



    data = pickle.load(open('Data/Fire/Fire.pkl', 'rb'))
    for i in range(len(data)):
        data[i]['data'] = torch.as_tensor(data[i]['data'], dtype=torch.float32).unsqueeze(-1)


    return data

if __name__ == '__main__':
    data = cylinder()
    print("length of the final data:", len(data))
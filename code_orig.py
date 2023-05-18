import torch
from torch import nn, optim
from torch.nn import functional as F
import argparse
import datetime
import os
from copy import deepcopy
from tqdm import tqdm 
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
#import wandb
import torchvision.utils as vutils
from statsmodels.tsa.seasonal import STL
from scipy.signal import medfilt2d

def DataBatch(data, label, text, batchsize, shuffle=True):
    
    n = data.shape[0]
    if shuffle:
        index = np.random.permutation(n)
    else:
        index = np.arange(n)
    for i in range(int(np.ceil(n/batchsize))):
        inds = index[i*batchsize : min(n,(i+1)*batchsize)]
        yield data[inds], label[inds], text[inds]

def angular_velocities(q1, q2, dt):
    x1, y1, z1, w1 = torch.split(q1, 1, dim=-1)
    x2, y2, z2, w2 = torch.split(q2, 1, dim=-1)
    x = w1*x2 - x1*w2 - y1*z2 + z1*y2
    y = w1*y2 + x1*z2 - y1*w2 - z1*x2
    z = w1*z2 - x1*y2 + y1*x2 - z1*w2
    w = (2 / dt) * torch.cat((x,y,z), dim=-1)
    return w
 
def rotation_matrix_from_quat(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    
    q1, q2, q3, q0 = torch.split(Q, 1, dim=-1)
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    r0 = torch.cat((r00, r01, r02), dim=-1)
    r1 = torch.cat((r10, r11, r12), dim=-1)
    r2 = torch.cat((r20, r21, r22), dim=-1)
    rot_matrix = torch.stack((r0, r1, r2), dim=-2)
                            
    return rot_matrix

def compute_w(vi, imu, mat_r=None):
    w_from_quat = angular_velocities(vi[:,:-1,-4:], vi[:,1:,-4:], 0.01) #b,t,3
    end = w_from_quat[:,-1,:].unsqueeze(1)
    w_from_quat = torch.cat((w_from_quat, end), axis=1)
    return w_from_quat

def compute_a(vi, imu, mat_r=None):
    velocity_vi_global = (vi[:,1:,:3] - vi[:,:-1,:3]) / 0.01
    accel_vi_global = (velocity_vi_global[:,1:] - velocity_vi_global[:,:-1]) / 0.01 
    end = accel_vi_global[:,-1,:].unsqueeze(1)
    accel_vi_global = torch.cat((accel_vi_global, end, end), axis=1)
    accel_vi_global[:,:,-1] += 9.8

    accel_vi_local = torch.einsum('btji,bti->btj', \
        rotation_matrix_from_quat(vi[:,:,-4:]).permute(0,1,3,2), accel_vi_global) #btij -> btji,bti -> btj
   
    return accel_vi_local

def sample_noise(data, rate):
    noise = 0.1 * torch.randn((data.shape[0], data.shape[1], data.shape[2])).to(device)
   
    idx1 = np.random.choice(data.shape[1], int(rate*data.shape[1]))
    noise_data = deepcopy(data)
    noise_data[:,idx1] = data[:,idx1] + noise[:,idx1]

    idx3 = np.random.choice(range(1,data.shape[1]), int(rate*(data.shape[1]-1)))
    noise_data[:,idx3] = noise_data[:,idx3-1]

    return noise_data, np.concatenate((idx1,idx3))


class AutoEncoder(nn.Module):
    def __init__(self, in_dim, z_dim, out_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=in_dim, out_channels=128, kernel_size=7, padding=3, padding_mode='replicate')

        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding=2, padding_mode='replicate')

        self.conv3 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1, padding_mode='replicate')

        self.conv4 = nn.Conv1d(in_channels=128, out_channels=out_dim, kernel_size=3, padding=1, padding_mode='replicate')
    
    def forward(self, vi, imu_w, imu_a):

        input = torch.cat((vi, imu_w, imu_a), dim=-1)
        b,t,c = input.size()

        x = F.relu((self.conv1(input.permute(0,2,1))))
        x = F.relu((self.conv2(x)))
        x = F.relu((self.conv3(x))) #b,128,t
        out = (self.conv4(x)).permute(0,2,1) #b,c,t
        
        return out
    

z = 10
epochs = 5001
bs = 16
seq_len = 3000
logdir = '/log'
model_path = '/checkpoint'
run_tag = ''
train = 1
lam = 0.5 
rate = 0.01
batchSize = 16 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   


# load dataset
high_vi = torch.from_numpy(np.load('./data/train/high_vi_%d.npy'%seq_len)).float().to(device)
high_w = torch.from_numpy(np.load('./data/train/high_w_%d.npy'%seq_len)).float().to(device)
high_a = torch.from_numpy(np.load('./data/train/high_a_%d.npy'%seq_len)).float().to(device)

low_vi = torch.from_numpy(np.load('./data/test/low_vi_%d.npy'%seq_len)).float().to(device)
low_w = torch.from_numpy(np.load('./data/test/low_w_%d.npy'%seq_len)).float().to(device)
low_a = torch.from_numpy(np.load('./data/test/low_a_%d.npy'%seq_len)).float().to(device)

seq_len, vi_dim, imu_w_dim, imu_a_dim = high_vi.shape[1], high_vi.shape[2], high_w.shape[2], high_a.shape[2]


# define model
model = AutoEncoder(in_dim=vi_dim+imu_w_dim+imu_a_dim, z_dim=z, out_dim=vi_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss().to(device)

# initial difference
vi_w = compute_w(high_vi, high_w)
vi_a = compute_a(high_vi, high_a)

init_w_loss = criterion(vi_w, high_w).item()
init_a_loss = criterion(vi_a, high_a).item()

print('initial w_loss: %.4f a_loss: %.4f' % (init_w_loss, init_a_loss))

vi_w = compute_w(low_vi, low_w)
vi_a = compute_a(low_vi, low_a)

init_w_loss = criterion(vi_w, low_w).item()
init_a_loss = criterion(vi_a, low_a).item()
print('initial w_loss: %.4f a_loss: %.4f' % (init_w_loss, init_a_loss))

if not os.path.isdir('vis_'+run_tag):
    os.mkdir('vis_'+run_tag)

print('log')
'''
# pre-train      
if train == 1:

    best_loss, best_val_loss = 10000, 10000
    for epoch in range(100):

        total_loss, total_vi_loss, total_w_loss, total_a_loss = 0, 0, 0, 0
        for batch_vi, batch_w, batch_a in DataBatch(high_vi, high_w, high_a, batchSize):
            
            noise_vi, idx1 = sample_noise(batch_vi, rate)
            denoise_vi = model(noise_vi, batch_w, batch_a)

            loss = criterion(denoise_vi[:,idx1], batch_vi[:,idx1])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += len(batch_vi) * loss.item()

        print('[%d] pretrain total_loss: %.4f vi_loss: %.4f w_loss: %.4f a_loss: %.4f' % \
            (epoch, total_loss/len(high_vi), total_vi_loss/len(high_vi), total_w_loss/len(high_vi), total_a_loss/len(high_vi)))
        
        if total_loss/len(high_vi) < best_loss:
            best_loss = total_loss/len(high_vi)
            torch.save(model.state_dict(), model_path + run_tag + 'pretrain_model')

    model.load_state_dict(torch.load(model_path + run_tag + 'pretrain_model'))
'''
# train and validation
best_loss = 10000
for epoch in range(epochs):

    total_loss, total_vi_loss, total_w_loss, total_a_loss = 0, 0, 0, 0
    for batch_vi, batch_w, batch_a in DataBatch(high_vi, high_w, high_a, batchSize):

        noise_vi, idx1 = sample_noise(batch_vi, rate)
        denoise_vi = model(noise_vi, batch_w, batch_a)

        vi_w = compute_w(denoise_vi, batch_w)
        vi_a = compute_a(denoise_vi, batch_a)

        vi_loss = criterion(denoise_vi[:,idx1], batch_vi[:,idx1]) 

        w_loss = criterion(vi_w, batch_w)
        a_loss = criterion(vi_a, batch_a)
        imu_loss = w_loss + a_loss

        w1 = float(vi_loss.item()/w_loss.item())
        w2 = float(vi_loss.item()/a_loss.item())
      
        loss = vi_loss + w1*w_loss + w2*a_loss 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += len(batch_vi) * loss.item()
        total_vi_loss += len(batch_vi) * vi_loss.item()
        total_w_loss += len(batch_vi) * w_loss.item()
        total_a_loss += len(batch_vi) * a_loss.item()

    if epoch % 1000 == 0:
        torch.save(model.state_dict(), model_path + run_tag + 'model' + str(epoch))

    print('[%d/%d] train total_loss: %.4f vi_loss: %.4f w_loss: %.4f a_loss: %.4f' % \
        (epoch, epochs, total_loss/len(high_vi), total_vi_loss/len(high_vi), total_w_loss/len(high_vi), total_a_loss/len(high_vi)))
    
    if total_loss < best_loss:
        best_loss = total_loss
        torch.save(model.state_dict(), model_path + run_tag + 'model')
#test
total_loss, total_vi_loss, total_w_loss, total_a_loss = 0, 0, 0, 0
denoise_vi_list, vi_w_list, vi_a_list = [], [], []
for batch_vi, batch_w, batch_a in DataBatch(low_vi, low_w, low_a, batchSize, shuffle=False):

    denoise_vi = model(batch_vi, batch_w, batch_a)

    vi_w = compute_w(denoise_vi, batch_w)
    vi_a = compute_a(denoise_vi, batch_a)

    denoise_vi_list.append(denoise_vi)
    vi_w_list.append(vi_w)
    vi_a_list.append(vi_a)

    vi_loss = criterion(denoise_vi, batch_vi)
    w_loss = criterion(vi_w, batch_w)
    a_loss = criterion(vi_a, batch_a)
    loss = w_loss + a_loss

    total_loss += len(batch_vi) * loss.item()
    total_vi_loss += len(batch_vi) * vi_loss.item()

denoise_vi = torch.cat(denoise_vi_list)
vi_w = torch.cat(vi_w_list)
vi_a = torch.cat(vi_a_list)

total_w_loss = criterion(vi_w, low_w).item()
total_a_loss = criterion(vi_a, low_a).item()

print('test total_loss: %.4f vi_loss: %.4f w_loss: %.4f a_loss: %.4f' % \
    (total_loss/len(low_vi), total_vi_loss/len(low_vi), total_w_loss, total_a_loss))



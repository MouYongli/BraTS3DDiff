import yaml
import torch
import numpy as np
from einops import rearrange
from matplotlib import pyplot as plt
from typing import Sequence
import pandas as pd

def compute_histogram(patch_vol_fracs, nbins=10, res=16):
    '''
    Compute histogram of tumor vol fracs of patches in an image.
    Count zero vol patches and non zero patches separately
    '''
    #patch_vol_fracs: (1,8,8,8) tensor
    patch_vol_fracs = patch_vol_fracs.view(1, -1)

    non_zero_vol_fracs = patch_vol_fracs[patch_vol_fracs > 0]
    hist = torch.histogram(input=non_zero_vol_fracs, bins=nbins, range=(non_zero_vol_fracs.min().item(),1))

    num_zero_patches = (patch_vol_fracs == 0).sum().item()
    bin_counts = torch.cat((torch.tensor([num_zero_patches]), hist.hist)).numpy()
    bin_edges = torch.cat((torch.tensor([0.0]), hist.bin_edges)).numpy()
    #print('bin_edges', bin_edges)
    bin_labels = ["0 (zero)"] + [
        f"{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}" for i in range(1, len(bin_edges) - 1)
    ]

    num_non_zero_patches = non_zero_vol_fracs.shape[0]
    counts = np.array([num_zero_patches, num_non_zero_patches])
    labels = ['Zero Vol Patches', 'Non-Zero Vol Patches']

    return {f'hist_{res}':(bin_counts, bin_labels), f'hist2_{res}':(counts, labels)}


def count_samples(wts,eps=0,k=32,N=1000):
    #count the number of times an item was sampled,
    # after sampling k out of len(wts) items N times
    counts = torch.zeros_like(wts)
    wts_ = wts + eps

    for i in range(N):
        sampled_indices = torch.multinomial(wts_,k)
        counts[sampled_indices] += 1
    return counts

def count_samples_dyn_eps(wts,start_eps=0.01,schedule='linear',k=32,N=1000,use_frac=0.8):
    counts = torch.zeros_like(wts)
    epss = []
    ps=[]
    for i in range(N):
        N_use = N*use_frac
        if i <= N_use - 1:
            if schedule == 'linear':
                eps = start_eps*(1-(i/N_use))
            elif schedule == 'cosine':
                eps = start_eps*np.cos((np.pi*i)/(2*N_use))
        epss.append(eps)
        wts_ = wts + eps
        sampled_indices = torch.multinomial(wts_,k)
        counts[sampled_indices] += 1
    return counts, epss



if __name__== "__main__":
    rootdir = '/home/ca550013/Master-Thesis/Projects/BraTS3DDiff'
    from src.datasets.patch_tumor_classify_diffusion_datamodule_new_new import BraTSDataModule

    with open(f"{rootdir}/configs/data/brats23_patch_tumor_classify_diffusion_new.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg.pop('_target_')
    cfg['data_dir'] = f"{rootdir}/data/BraTS-Data/BraTS2023-GLI"
    cfg['batch_size'] = 1
    cfg['patch_sizes'] = [16, 32]
    datamodule = BraTSDataModule(**cfg)
    train_data = datamodule.data_train
    test_data = datamodule.data_test
    val_data = datamodule.data_val

    '''
    Compute patch tumor vol distributions of all images in dataset, and compute some statistics
    '''
    hist_data = {'hist_16':[], 'hist2_16':[], 'hist_32':[], 'hist2_32':[]}
    patch_tumor_vol_fracs = {'16':[], '32': []}
    for idx in range(len(val_data)):
        data = val_data.__getitem__(idx)
        patch_tumor_vol_fracs['16'].append(data['patch_tumor_vol_fracs']['16'])
        hist_16 = compute_histogram(data['patch_tumor_vol_fracs']['16'], res=16)
        hist_data['hist_16'].append(hist_16['hist_16'][0])
        hist_data['hist2_16'].append(hist_16['hist2_16'][0])
        patch_tumor_vol_fracs['32'].append(data['patch_tumor_vol_fracs']['32'])
        hist_32 = compute_histogram(data['patch_tumor_vol_fracs']['32'], res=32)
        hist_data['hist_32'].append(hist_32['hist_32'][0])
        hist_data['hist2_32'].append(hist_32['hist2_32'][0])


    mean_hist = {}
    median_hist = {}
    max_hist = {}
    min_hist = {}

    for key in hist_data.keys():
        hist_data[key] = np.array(hist_data[key])
        mean_hist[key] = np.mean(hist_data[key], axis=0)
        median_hist[key] = np.median(hist_data[key], axis=0)
        max_hist[key] = np.max(hist_data[key], axis=0)
        min_hist[key] = np.min(hist_data[key], axis=0)

    print('Mean Statistics')
    print(mean_hist)
    print('Median Statistics')
    print(median_hist)
    print('Max Statistics')
    print(max_hist)
    print('Min Statistics')
    print(min_hist)


    N=300 #num of sampling rounds
    k=256  #num of patches to sample out of 256 patches

    base_epss = [20,0.1,0.01,0.001,1e-4,1e-20]
    use_fracs = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
    schedules = ['linear','cosine']
    start_epss_1 = np.around(np.linspace(0.1, 0.01, 10), decimals=2).tolist()
    start_epss_2 = np.around(np.linspace(1.0, 0.2, 9), decimals=2).tolist()
    start_epss = start_epss_1 + start_epss_2 +[0.001, 1e-4]

    print('start_epss', start_epss)

    fixed_eps_df_rows = []
    dyn_eps_df_rows = []
    key = 'hist2_16'
    for idx in range(len(val_data)):
        print(f'Processing image {idx}')
        print('Non-Tumor and Tumor Patch Counts',hist_data[key][idx])
        num_tumor = hist_data[key][idx][1]
        #get patch tumor vols of an image and reshape into 1D tensor
        all_patch_vols = patch_tumor_vol_fracs['16'][idx].view(1,-1)
        all_patch_vols=torch.sort(all_patch_vols).values
        print('All patches vol fracs sorted' ,all_patch_vols)
        print('Tumor Patch indices' ,all_patch_vols.nonzero()[:,1])
        all_patch_vols=all_patch_vols.squeeze(dim=0)

        #uniform and weighted sampling with varying eps
        cmp_counts_weighted = count_samples(all_patch_vols,eps=base_epss[-1],k=k,N=N)
        cmp_counts_weighted_p = cmp_counts_weighted/cmp_counts_weighted.sum()
        cmp_counts_uniform = count_samples(all_patch_vols,eps=base_epss[0],k=k,N=N)
        cmp_counts_uniform_p = cmp_counts_uniform/cmp_counts_uniform.sum()

        uniform_wts = torch.ones_like(all_patch_vols)/len(all_patch_vols)
        tumor_wts = all_patch_vols/all_patch_vols.sum()

        #Fixed eps
        for i,eps in enumerate(base_epss):
            fixed_eps_df_row = {'image_id':idx, 'num_tumor_patches':num_tumor}
            fixed_eps_df_row['eps'] = eps
            counts = count_samples(all_patch_vols,eps=eps,k=k,N=N)
            counts_p = counts/counts.sum()
            kl_wt_1 = torch.nn.functional.kl_div(counts_p.log(), cmp_counts_weighted_p).item()
            kl_uni_1 = torch.nn.functional.kl_div(counts_p.log(), cmp_counts_uniform_p).item()
            score_1 = 10*(kl_wt_1+kl_uni_1)
        
            kl_wt_2 = torch.nn.functional.kl_div(counts_p.log(), tumor_wts).item()
            kl_uni_2 = torch.nn.functional.kl_div(counts_p.log(), uniform_wts).item()
            score_2 = kl_wt_2+kl_uni_2
            mean_score = 0.5*(score_1+score_2)

            fixed_eps_df_row.update({'kl_wt_1':kl_wt_1,'kl_uni_1':kl_uni_1,'score_1':score_1, \
                            'kl_wt_2':kl_wt_2,'kl_uni_2':kl_uni_2,'score_2':score_2, 'mean_score':mean_score})
            fixed_eps_df_rows.append(fixed_eps_df_row)

        #Dynamic p and eps
        for use_frac in use_fracs:
            for schedule in schedules:
                for i,start_eps in enumerate(start_epss):
                    dyn_eps_df_row = {'image_id':idx, 'num_tumor_patches':num_tumor}
                    dyn_eps_df_row.update({'schedule':schedule, 'use_frac':use_frac, 'start_eps':start_eps})
                    
                    counts, epss_= count_samples_dyn_eps(all_patch_vols,start_eps=start_eps,schedule=schedule,k=k,N=N,use_frac=use_frac)
                    counts_p = counts/counts.sum()

                    kl_wt_1 = torch.nn.functional.kl_div(counts_p.log(), cmp_counts_weighted_p).item()
                    kl_uni_1 = torch.nn.functional.kl_div(counts_p.log(), cmp_counts_uniform_p).item()
                    score_1 = 10*(kl_wt_1+kl_uni_1)
                
                    kl_wt_2 = torch.nn.functional.kl_div(counts_p.log(), tumor_wts).item()
                    kl_uni_2 = torch.nn.functional.kl_div(counts_p.log(), uniform_wts).item()
                    score_2 = kl_wt_2+kl_uni_2
                    mean_score = 0.5*(score_1+score_2)

                    dyn_eps_df_row.update({'kl_wt_1':kl_wt_1,'kl_uni_1':kl_uni_1,'score_1':score_1, \
                                    'kl_wt_2':kl_wt_2,'kl_uni_2':kl_uni_2,'score_2':score_2, 'mean_score':mean_score})
                    dyn_eps_df_rows.append(dyn_eps_df_row)


    fixed_eps_df = pd.DataFrame.from_dict(fixed_eps_df_rows)
    dyn_eps_df = pd.DataFrame.from_dict(dyn_eps_df_rows)
    dyn_eps_df.to_csv('dyn_eps_df_1.csv')
    fixed_eps_df.to_csv('fixed_eps_df_1.csv')
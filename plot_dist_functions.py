import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

def plot_dist(feature, pooled_df1, pooled_df2, df1_label, df2_label, save_path, plot_name):
    

    num_bins = int(np.sqrt(len(pooled_df1[feature])))
    
    hist, bin_edges = np.histogram(pooled_df1[feature].dropna(),bins=num_bins)
    
    plt.hist(pooled_df1[feature].dropna(),bins=bin_edges,histtype='stepfilled',color=(128/256,0,0),label='{}'.format(df1_label),density=True,alpha=0.5)
    plt.hist(pooled_df2[feature].dropna(),bins=bin_edges,histtype='stepfilled',color=(0,102/256,102/256),label='{}'.format(df2_label),density=True,alpha=0.5)
    plt.title('{}'.format(plot_name))
    plt.xlabel('{}'.format(feature))
    plt.ylabel('density')
    plt.legend()

    plt.savefig('{}/{}.pdf'.format(save_path, plot_name),bbox_inches='tight',format='pdf')
    plt.clf()

def plot_dist_onetype(feature, pooled_df, df_label, save_path, plot_name):
    
    if df_label == 'WT':
        c = (128/256,0,0)
    else:
        c = (0,102/256,102/256)
    
    plt.hist(pooled_df[feature].dropna(),bins='auto',histtype='stepfilled',color=c,label='{}'.format(df_label),density=True)
    plt.title('{}'.format(plot_name))
    plt.xlabel('{}'.format(feature))
    plt.ylabel('density')
    plt.legend()

    plt.savefig('{}/{}.pdf'.format(save_path, plot_name),bbox_inches='tight',format='pdf')
    plt.clf()

def plot_dist_all(feature, pooled_df1, pooled_df2, pooled_df3, pooled_df4, df1_label, df2_label, df3_label, df4_label, save_path, plot_name):
    
    # arr = arpc2ko_df[param].to_numpy()
    # arr = arr[~np.isnan(arr)]
    
    hist, bin_edges = np.histogram(pooled_df1[feature].dropna(),bins=50)
    
    plt.hist(pooled_df1[feature].dropna(),bins=bin_edges,histtype='step',color='blue',label='{}'.format(df1_label),density=True)
    plt.hist(pooled_df2[feature].dropna(),bins=bin_edges,histtype='step',color='red',label='{}'.format(df2_label),density=True)
    plt.hist(pooled_df3[feature].dropna(),bins=bin_edges,histtype='step',color='green',label='{}'.format(df3_label),density=True)
    plt.hist(pooled_df4[feature].dropna(),bins=bin_edges,histtype='step',color='orange',label='{}'.format(df4_label),density=True)
    # plt.title('{}'.format(feature))
    plt.title('{}'.format(plot_name))
    plt.xlabel('{}'.format(feature))
    plt.ylabel('density')
    plt.legend()

    plt.savefig('{}/{}.png'.format(save_path, plot_name),bbox_inches='tight')
    plt.clf()
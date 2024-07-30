import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from glob import glob
import os
import warnings

warnings.filterwarnings('ignore')

def get_metrics(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    metrics = checkpoint['metrics']
    fname = '/'.join(ckpt_path.split('/')[:-1])
    return metrics, fname

def plot_curves(paths, ylog=True, xlog=False, xmin=None, xmax=None, offset = 0.02, plot_metrics=True):
    if plot_metrics:
        fig, axes = plt.subplots(3,1,figsize=(10,5*1.5), dpi=150, sharex=True)
        ax_loss, ax_RMSD, ax_HDD = axes
    else:
        fig, axes = plt.subplots(figsize=(10,5), dpi=150)
        ax_loss = axes
        axes = [axes]

    text_positions = []
    
    if plot_metrics:
        ax_loss.set( ylabel='CE-Loss')
        ax_RMSD.set(ylabel='Mean RMSD')
        ax_HDD.set(xlabel='Epoch', ylabel='Mean Hausdorff distance')
    else:
        ax_loss.set(xlabel='Epoch', ylabel='CE-Loss')

    for ax in axes:
        if xmin is not None or xmax is not None:
            ax.set_xlim(xmin, xmax)
        if ylog:
            ax_loss.set_yscale('log')
        if xlog:
            ax_loss.set_xscale('log')
        ax.grid(alpha=0.2, which="both")

    for path in paths:
        metrics, fname = get_metrics(path)
        losses_train, losses_val, epochs = metrics['train_losses'], metrics['val_losses'], metrics['epoch_losses']
        RMSD_train, RMSD_val, RMSD_epochs = metrics['train_RMSD'], metrics['val_RMSD'], metrics['epoch_RMSD']
        HDD_train, HDD_val, HDD_epochs = metrics['train_HDD'], metrics['val_HDD'], metrics['epoch_HDD']
        
        # Losses
        p = ax_loss.plot(epochs, losses_train, label=fname + f' [{losses_val[-1].item():1.3f}]')
        ax_loss.plot(epochs, losses_val, c=p[0].get_color(), ls='--')
        
        if plot_metrics:
            # RMSD
            p = ax_RMSD.plot(RMSD_epochs, RMSD_train, label=fname)
            ax_RMSD.plot(RMSD_epochs, RMSD_val, c=p[0].get_color(), ls='--')

            # HDD
            p = ax_HDD.plot(HDD_epochs, HDD_train, label=fname)
            ax_HDD.plot(HDD_epochs, HDD_val, c=p[0].get_color(), ls='--')
        
        # Find the minimum value in losses_val and its corresponding epoch
        try:
            val_line_min = epochs[np.argmin(losses_val)].item()
            min_loss_val = torch.min(losses_val).item()
        
            # Plot the dotted line
            ax_loss.plot([val_line_min, ax.get_xlim()[1]], [min_loss_val, min_loss_val],
                        c=p[0].get_color(), ls=':', alpha=1.0)

            # Adjust text position if overlapping
            text_x = ax.get_xlim()[1]
            text_y = min_loss_val
        
            vert_align = 'bottom'
            for pos in text_positions:
                if abs(pos[1] - text_y) < offset:  # Check for overlap
                    vert_align = 'top'
                else:
                    vert_align = 'bottom'

            # Add text at the end of the dotted line
            ax_loss.text(text_x, text_y, f'{min_loss_val:.4f}', 
                    verticalalignment=vert_align, horizontalalignment='right', color=p[0].get_color(),
                    fontsize=10)
            text_positions.append((text_x, text_y))
        except:
            pass
     
    for ax in axes:
        ax.legend(fontsize=8, ncol=4)
    fig.tight_layout()
    plt.show()
    
    
def plot_disitributions(eval_path, filter_entries_above=0, kde=True, stat='density', log_scale=False, bins=61):
    
    # Create the subplots
    fig, axs = plt.subplots(2, 1, figsize=(14, 12))
    
    ## VALID DISTRIBUTION ##
    
    data = pd.read_csv(eval_path)
    all_entries = data['gen_len']
    valid_entries = data[data['is_valid'] == True]['gen_len']
    mean_all_entries = all_entries.mean()
    mean_valid_entries = valid_entries.mean()
    count_all_entries = len(all_entries)
    count_valid_entries = len(valid_entries)
    
    # Determine the binning range with smaller bin size
    min_gen_len = min(all_entries.min(), valid_entries.min())
    max_gen_len = max(all_entries.max(), valid_entries.max())
    bins_half_size = np.linspace(min_gen_len, max_gen_len, bins)
    
    # Distribution of gen_len for data
    sns.histplot(all_entries, kde=kde, color='blue', label=f'All Entries (N={count_all_entries})', bins=bins_half_size, stat=stat, ax=axs[0])
    sns.histplot(valid_entries, kde=kde, color='green', label=f'Valid Entries (N={count_valid_entries})', bins=bins_half_size, stat=stat, ax=axs[0])
    axs[0].axvline(mean_all_entries, color='blue', linestyle='--', linewidth=2, label=f'Mean All Entries: {mean_all_entries:.2f}')
    axs[0].axvline(mean_valid_entries, color='green', linestyle='--', linewidth=2, label=f'Mean Valid Entries: {mean_valid_entries:.2f}')
    axs[0].set_ylabel(stat)
    axs[0].legend()
    axs[0].set_yticklabels([])
    axs[0].grid(alpha=0.2)
    if log_scale:
        axs[0].set_yscale('log')
    
    ## SPACEGROUP DISTRIBUTION ##
    data = pd.read_csv(eval_path)

    # Calculate the percent of invalid entries for each spacegroup in both datasets
    spacegroups_all = data['sg'].value_counts()
    spacegroups_invalid = data[data['is_valid'] == True]['sg'].value_counts()
    invalid_percent = (spacegroups_invalid / spacegroups_all * 100).dropna()

    # Filter
    invalid_percent = invalid_percent[spacegroups_all > filter_entries_above]

    # Create dataframes for visualization
    invalid_df = invalid_percent.reset_index()
    invalid_df.columns = ['Spacegroup', 'Percent Valid']

    # Plot the percent invalid for each spacegroup
    sns.barplot(data=invalid_df, x='Spacegroup', y='Percent Valid', palette='viridis', ax=axs[1])
    axs[1].set_title('Percent Valid Spacegroups')
    axs[1].set_xlabel('Spacegroup')
    axs[1].set_ylabel('Percent Valid')
    axs[1].tick_params(axis='x', rotation=90)

    plt.tight_layout()
    plt.show()

#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--output',required=True); ap.add_argument('--title',required=True); args=ap.parse_args()
    d=pd.read_csv(args.input)
    if 'mean' in d.columns:
        g=d.copy()
    elif 'correct_information_gain_bits' in d.columns:
        g=d.rename(columns={'correct_information_gain_bits':'mean'})
    else:
        g=d.groupby('pattern')['correct_information_bits'].agg(['mean','sem']).reset_index()
    g=g.sort_values('pattern')
    if 'sem' not in g.columns: g['sem']=0
    fig,ax=plt.subplots(figsize=(7.0,5.0))
    x=range(len(g))
    ax.errorbar(x,g['mean'],yerr=g['sem'].fillna(0),color='#7B3FB2',marker='o',markersize=10,markerfacecolor='#7B3FB2',markeredgecolor='black',markeredgewidth=2.0,linewidth=4.2,elinewidth=2.4,capsize=6,capthick=2.4,zorder=3)
    ax.set_xticks(list(x)); ax.set_xticklabels(g.pattern.astype(str))
    ax.set_xlabel('Specific Feature Enrichment',fontsize=20,fontweight='bold',labelpad=12)
    ax.set_ylabel('information gain (bits)',fontsize=20,fontweight='bold',labelpad=12)
    ax.tick_params(axis='both',labelsize=18,width=2.4,length=8,direction='out')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2.6); ax.spines['bottom'].set_linewidth(2.6)
    ax.spines['left'].set_position(('outward',12)); ax.spines['bottom'].set_position(('outward',12))
    ax.spines['bottom'].set_bounds(0,max(x)); ax.grid(False)
    ax.set_xlim(-.28,max(x)+.28)
    fig.subplots_adjust(left=.24,right=.96,bottom=.24,top=.94)
    out=Path(args.output); fig.savefig(out.with_suffix('.png'),dpi=240,bbox_inches='tight'); fig.savefig(out.with_suffix('.svg'),bbox_inches='tight'); plt.close(fig)
if __name__=='__main__': main()

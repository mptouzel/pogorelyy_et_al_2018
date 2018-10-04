# pogorelyy_et_al_2018
code associated to publication

## Description
Code producing tables of significantly expanded clones from Pogorely et al. "Precise tracking of vaccine-responding T-cell clones reveals convergent and personalized response in identical twins" 
N.B. it is presented here for the review process only. A user-oriented version can be found with Puelma Touzel et al. (in preparation).

## Contents
Code consists of a main python script (infer_diff_main.py), which makes use of functions contained in a function library file (infer_diff_lib.py). 

## Instructions to install and run code on an example
1. install python 2.7 (https://www.python.org/download/releases/2.7/) on your system with the following package dependecies: *numpy*, *scipy*, and *pandas*. This is accomplished most easily by installing a python scientific computing distribution, e.g. anaconda: https://www.anaconda.com/download/), which already comes with these packages. On linux, this amounts to downloadling and running the installer:
> `bash Anaconda-latest-Linux-x86_64.sh`
2. download the repository and unarchive Yellow fever/Yellow_fever_S1_example test folder
3. run the following command in the repository rootpath
> `python infer_diffexpr_main.py S1_0_F1_.txt S1_0_F2_.txt S1_0_F1_.txt S1_15_F1_.txt`
    The first two arguments are the paths to the data files of the desired null model pair: biological replicates of day 0; the third and fourth arguments are the first and second time point to be compared. This produces the results for a day0-day15 comparison using the first replicate of both, and using the day 0-replicate 1 and day 0-replicate 2 null model.
4. Final output is located in outdata/S1_0_F1_S1_15_F1/min0_maxinf_S1_0_F1_S1_0_F2_v1_manu_version/S1_0_F1_S1_15_F1_tabletop_expanded.csv

## Script behavior and output
This main script reads in files from a folder ('Yellow_fever') containing the processed data of RNA molecule counts and produces a folder structure (with parent folder 'outdata') into which the output data is written, including the final result as a .csv file containing the list of expanded clones with their features. Downstream analysis extracts the subset of this list by selecting clones having the non-expansion probability, _1-P(s>0)_, below 0.05 and the median _s_, _s{med}_, greater than _\log_e(2^5)_ (approximately _3.46_).The data in 'Yellow fever' has been been compressed to faciliate download of the code. 

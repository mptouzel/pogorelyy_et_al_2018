# pogorelyy_et_al_2018
code associated to publication

## Description
Code producing tables of significantly expanded clones from Pogorely et al. "Precise tracking of vaccine-responding T-cell clones reveals convergent and personalized response in identical twins" 
N.B. It is presented here for the review process only. A user-oriented version can be found with Puelma Touzel et al. (in preparation).

## Contents
Code consists of a main python script (infer_diff_main.py), which makes use of functions contained in a function library file (infer_diff_lib.py). 

## Dependencies
The functions make use of the numpy, scipy, and pandas packages already shipped with many python scientific computing distributions (e.g. anaconda). 

## Script usage and behavior
This main script reads in files from a folder ('Yellow_fever') containing the processed data of RNA molecule counts and produces a folder structure (with parent folder 'outdata') into which the output data is written, including the final result as a .csv file containing the list of expanded clones with their features. Downstream analysis extracts the subset of this list by selecting clones having the non-expansion probability, $1-P(s>0)$, below 0.05 and the median $s$, $s_{med}$, greater than $\log_e(2^5)$ (approximately $3.46$).The data in 'Yellow fever' has been been compressed (to 'Yellow_fever.tar.gz') to faciliate download of the code. 

The main script takes 9 command-line input arguments specifiying the datasets on which to learn. The first argument is the name of the desired donor. The next 8 arguments are integers corresponding to the 4 indices specifying the two data sets (each specified by their day and replicate index) used for null model, and then the 4 indices from the pair of data sets to be compared. The day vector is $(pre0, 0,7,15,45)$, and the replicate vector is $(1,2)$. 

## Example
We have pulled out an example dataset (day 0 and day 15 of donor S1) in the Yellow_fever folder so that running

python infer_diff_main.py S1 1 0 1 1 1 0 3 0

produces the results for a day0-day15 comparison using the first replicate of both (1 0 3 0), and using the day 0-replicate 1 and day 0-replicate 2 null model (1 0 1 1).

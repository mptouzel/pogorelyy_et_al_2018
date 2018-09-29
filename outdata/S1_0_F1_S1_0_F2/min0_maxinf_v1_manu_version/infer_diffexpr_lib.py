#!/home/max/anaconda2/bin
import numpy as np
import math
from matplotlib import use
from copy import deepcopy
import pandas as pd
import scipy.stats
import pylab as pl
from pylab import rcParams
from functools import partial

def NegBinPar(m,v,mvec): #speed up only insofar as the log and exp are called once on array instead of multiple times on rows
    mmax=mvec[-1]
    p = 1-m/v
    r = m*m/v/p
    NBvec=np.arange(mmax+1,dtype=float)   
    NBvec[1:]=np.log((NBvec[1:]+r-1)/NBvec[1:]*p) #vectorization won't help unfortuneately here since log needs to be over array
    NBvec[0]=r*math.log(m/v)
    NBvec=np.exp(np.cumsum(NBvec)[mvec]) #save a bit here
    return NBvec
  
def PoisPar(Mvec,unicountvals):
    #assumes Mvec starts at 0; unicountvals doesn't have to start at 0
    nmax=unicountvals[-1]
    nlen=len(unicountvals)
    mlen=len(Mvec)
    Nvec=unicountvals
    
    Nmtr=np.tile(np.reshape(Nvec,(1,nlen)),(mlen,1) )
    logNmtr=np.tile(np.reshape(-np.insert(np.cumsum(np.log(np.arange(1,nmax+1))),0,0.)[unicountvals],(1,nlen)),(mlen,1)) #avoid n=0 nans, could use broadcasting here
    Mmtr=np.tile(-np.reshape(Mvec,(mlen,1)),(1,nlen))
    logMmtr=np.tile(np.reshape(np.log(Mvec),(mlen,1)),(1,nlen))#throws warning: since log(0)=-inf  
    
    Nmtr=np.exp(Nmtr*logMmtr+logNmtr+Mmtr) #throws warning: multiply by -inf
    if Mvec[0]==0:
        Nmtr[0,:]=np.zeros((nlen,)) #when m=0, n=0, and so get rid of nans from log(0)
        Nmtr[0,0]=1. #handled below
    if unicountvals[0]==0: #if n=0 included get rid of nans from log(0)
        Nmtr[:,0]=np.exp(-Mvec)
    return Nmtr
  
def get_modelsample(pmf,Nsamp):
    shape = np.shape(pmf)
    sortindex = np.argsort(pmf, axis=None)#uses flattened array
    pmf = pmf.flatten()
    pmf = pmf[sortindex]
    cmf = np.cumsum(pmf)
    choice = np.random.uniform(high = cmf[-1], size = int(float(Nsamp)))
    index = np.searchsorted(cmf, choice)
    index = sortindex[index]
    index = np.unravel_index(index, shape)
    index = np.transpose(np.vstack(index))
    sampled_pairs = index[np.argsort(index[:,0])]
    return sampled_pairs
  
def get_sparserep(counts):
    counts['paircount']=1
    clone_counts=counts.groupby(['Clone_count_1','Clone_count_2']).sum()
    clonecountpair_counts=np.asarray(clone_counts.values.flatten(),dtype=int)
    clonecountpair_vals=clone_counts.index.values
    indn1=np.asarray([clonecountpair_vals[it][0] for it in range(len(clonecountpair_counts))],dtype=int)
    indn2=np.asarray([clonecountpair_vals[it][1] for it in range(len(clonecountpair_counts))],dtype=int)
    NreadsI=counts.Clone_count_1.sum()
    NreadsII=counts.Clone_count_2.sum()

    unicountvals_1,indn1=np.unique(indn1,return_inverse=True)
    unicountvals_2,indn2=np.unique(indn2,return_inverse=True)
  
    return indn1,indn2,clonecountpair_counts,unicountvals_1,unicountvals_2,NreadsI,NreadsII

def import_data(path,filename1,filename2,mincount,maxcount,colnames1,colnames2,headerline1,headerline2):
    columns2use1=colnames1 
    columns2use2=colnames2
    newnames=['Clone_fraction','Clone_count','ntCDR3','AACDR3']    
    with open(path+filename1, 'r') as f:
        F1Frame_chunk=pd.read_csv(f,delimiter='\t',usecols=columns2use1,header=headerline1)[columns2use1]
    with open(path+filename2, 'r') as f:
        F2Frame_chunk=pd.read_csv(f,delimiter='\t',usecols=columns2use2,header=headerline2)[columns2use2]
    F1Frame_chunk.columns=newnames
    F2Frame_chunk.columns=newnames
    suffixes=('_1','_2')
    mergedFrame=pd.merge(F1Frame_chunk,F2Frame_chunk,on=newnames[2],suffixes=suffixes,how='outer')
    for nameit in [0,1]:
        for labelit in suffixes:
            mergedFrame.loc[:,newnames[nameit]+labelit].fillna(int(0),inplace=True)
            if nameit==1:
                mergedFrame.loc[:,newnames[nameit]+labelit].astype(int)
    def dummy(x):
	val=x[0]
	if pd.isnull(val):
	    val=x[1]    
	return val
    mergedFrame.loc[:,newnames[3]+suffixes[0]]=mergedFrame.loc[:,[newnames[3]+suffixes[0],newnames[3]+suffixes[1]]].apply(dummy,axis=1) #combines strings in both columns, creates duplicates
    mergedFrame.drop(newnames[3]+suffixes[1], 1,inplace=True)
    mergedFrame.rename(columns = {newnames[3]+suffixes[0]:newnames[3]}, inplace = True)
    mergedFrame=mergedFrame[[newname+suffix for newname in newnames[:2] for suffix in suffixes]+[newnames[2],newnames[3]]]
    filterout=((mergedFrame.Clone_count_1<mincount) & (mergedFrame.Clone_count_2==0)) | ((mergedFrame.Clone_count_2<mincount) & (mergedFrame.Clone_count_1==0)) #has effect only if mincount>0
    original_number_clones=len(mergedFrame)
    return original_number_clones,mergedFrame.loc[((mergedFrame.Clone_count_1<=maxcount) & (mergedFrame.Clone_count_2<=maxcount)) & ~filterout]
  
def get_rhof(alpha_rho,freq_nbins):
    fmin=1e-11 #Smallest frequency is of singleton from total number of lymphocytes in body, use 10^11.
    fmax=1e-0
    logfvec=np.linspace(np.log10(fmin),np.log10(fmax),freq_nbins+1)
    fvec=np.power(10,logfvec)
    rhovec=np.power(fvec[:-1],alpha_rho)       
    normconst=np.trapz(rhovec, x=fvec[:-1]) 
    rhovec/=normconst  
    return rhovec,fvec

def get_Ps(alp,sbar,smax,stp):
    lamb=-stp/sbar
    smaxt=round(smax/stp)
    s_zeroind=int(smaxt)
    Z=2*(np.exp((smaxt+1)*lamb)-1)/(np.exp(lamb)-1)-1
    Ps=alp*np.exp(lamb*np.fabs(np.arange(-smaxt,smaxt+1)))/Z
    Ps[s_zeroind]+=(1-alp)
    return Ps
  
#@profile
def get_Pn1n2_s(paras, svec, unicountvals_1, unicountvals_2, NreadsI, NreadsII, nfbins, repfac, indn1=None ,indn2=None,countpaircounts_d=None,f2s_step=None):    
    #svec determines which of 3 run modes is evaluated
    #1) svec is array => compute P(n1,n2|s),           output: Pn1n2_s,unicountvals_1,unicountvals_2,Pn1_f,fvec,Pn2_s,svec
    #2) svec=-1       => null model likelihood,        output: data-averaged loglikelihood
    #3) else          => compute null model, P(n1,n2), output: Pn1n2_s,unicountvals_1,unicountvals_2,Pn1_f,Pn2_f,fvec
      
    #writes a model distribution for a given pair count data set from which parameters have already been learned
    alpha = paras[0]
    m_total=float(np.power(10, paras[3])) 
    r_c1=NreadsI/m_total 
    r_c2=repfac*r_c1     
    beta_mv= float(np.power(10,paras[1]))
    alpha_mv=paras[2]
      
    rhofvec,fvec = get_rhof(paras[0],nfbins)
    maxcountvals_1=max(unicountvals_1)
    maxcountvals_2=max(unicountvals_2)
    n1itvec=range(len(unicountvals_1))
    n2itvec=range(len(unicountvals_2))

    nsigma=5.
    nmin=100.
    #for each n, get actual range of m to compute around n-dependent mean m
    m1_low=np.zeros((len(unicountvals_1),),dtype=int)
    m1_high=np.zeros((len(unicountvals_1),),dtype=int)
    for n1it,n1 in enumerate(unicountvals_1):
	mean_m=n1/r_c1
	dev=nsigma*np.sqrt(mean_m)
	m1_low[n1it]=int(mean_m-dev) if (mean_m>dev**2) else 0
	m1_high[n1it]=(mean_m+5*dev) if (n1>nmin) else int(10.*nmin/r_c1)
    m_cellmax1=np.max(m1_high)
    
    m2_low=np.zeros((len(unicountvals_2),),dtype=int)
    m2_high=np.zeros((len(unicountvals_2),),dtype=int)
    for n2it,n2 in enumerate(unicountvals_2):
	mean_m=n2/r_c2#r_c
	dev=nsigma*np.sqrt(mean_m)
	m2_low[n2it]=int(mean_m-dev) if (mean_m>dev**2) else 0
	m2_high[n2it]=int(mean_m+5*dev) if (n2>nmin) else int(10.*nmin/r_c2)
    m_cellmax2=np.max(m2_high)

    #across n, collect all in-range m
    m1vec_bool=np.zeros((m_cellmax1+1,),dtype=bool)
    for n1it in n1itvec:
	m1vec_bool[m1_low[n1it]:m1_high[n1it]+1]=True
    m1vec=np.arange(m_cellmax1+1)[m1vec_bool]
    m2vec_bool=np.zeros((m_cellmax2+1,),dtype=bool)
    for n2it in n2itvec:
	m2vec_bool[m2_low[n2it]:m2_high[n2it]+1]=True
    m2vec=np.arange(m_cellmax2+1)[m2vec_bool]
    #transform to in-range index
    for n1it in n1itvec:
	m1_low[n1it]=np.where(m1_low[n1it]==m1vec)[0][0]
	m1_high[n1it]=np.where(m1_high[n1it]==m1vec)[0][0]
    for n2it in n2itvec:
	m2_low[n2it]=np.where(m2_low[n2it]==m2vec)[0][0]
	m2_high[n2it]=np.where(m2_high[n2it]==m2vec)[0][0]
    
    Poisvec=PoisPar(m1vec*r_c1,unicountvals_1)
    mean_cell_counts_1=m_total*fvec[:-1]
    var_cell_counts_1=mean_cell_counts_1+beta_mv*np.power(mean_cell_counts_1,alpha_mv)
    #print("computing P(n1|f)")
    Pn1_f=np.zeros((len(fvec)-1,len(unicountvals_1)))
    for f_it in range(len(fvec)-1):
	NBvec=NegBinPar(mean_cell_counts_1[f_it],var_cell_counts_1[f_it],m1vec)
	for n1_it in n1itvec:
	    Pn1_f[f_it,n1_it]=np.dot(NBvec[m1_low[n1_it]:m1_high[n1_it]+1],Poisvec[m1_low[n1_it]:m1_high[n1_it]+1,n1_it]) 
	    
    d=np.diff(fvec[:-1])
    if isinstance(svec,np.ndarray):
        logf_step=np.diff(np.log(fvec))[0]
	smax=(len(svec)-1)/2
	logfmin=np.log(fvec[0 ])-f2s_step*smax*logf_step
	logfmax=np.log(fvec[-1])+f2s_step*smax*logf_step
	fvecwide=np.exp(np.linspace(logfmin,logfmax,len(fvec)-1+2*smax*f2s_step))
	
	print("computing P(n2|f)")
	Pn2_f=np.zeros((len(fvecwide),len(unicountvals_2)))
	mean_cell_counts_2=m_total*fvecwide
	var_cell_counts_2=mean_cell_counts_2+beta_mv*np.power(mean_cell_counts_2,alpha_mv)
	Poisvec2=PoisPar(m2vec*r_c2,unicountvals_2)
	for f_it in range(len(fvecwide)):
	    NBvec=NegBinPar(mean_cell_counts_2[f_it],var_cell_counts_2[f_it],m2vec)
	    for n2_it in n2itvec:
		Pn2_f[f_it,n2_it]=np.dot(NBvec[m2_low[n2_it]:m2_high[n2_it]+1],Poisvec2[m2_low[n2_it]:m2_high[n2_it]+1,n2_it]) 

	print('computing P(n1,n2|f,s)')
	Pn1n2_s=np.zeros((len(svec),len(unicountvals_1),len(unicountvals_2))) 
	for s_it,s in enumerate(svec):
	    for n1_it,n2_it in zip(indn1,indn2):
		integ=rhofvec*Pn2_f[f2s_step*s_it:(f2s_step*s_it+len(fvec[:-1])),n2_it]*Pn1_f[:,n1_it]/2.
		Pn1n2_s[s_it,n1_it,n2_it] = np.dot(d,(integ[1:] + integ[:-1]))# / 2.0)
	Pn0n0_s=np.zeros(svec.shape)
	for s_it,s in enumerate(svec):    
	    integ=Pn1_f[:,0]*Pn2_f[f2s_step*s_it:(f2s_step*s_it+len(fvec[:-1])),0]*rhofvec/2.
	    Pn0n0_s[s_it]=np.sum(d[np.newaxis,:]*(integ[1:]+integ[:-1]),axis=1)

	Pn2_s=0
        return Pn1n2_s,unicountvals_1,unicountvals_2,Pn1_f,fvec,Pn2_s,Pn0n0_s,svec
      
    elif svec==-1:
        #print("running obj func")
	mean_cell_counts_2=m_total*fvec[:-1]
	var_cell_counts_2=mean_cell_counts_2+beta_mv*np.power(mean_cell_counts_2,alpha_mv)
	Poisvec2=PoisPar(m2vec*r_c2,unicountvals_2) #overwrite. some duplication computation here, but fast enough so leave it.  
	Pn2_f=np.zeros((len(fvec)-1,len(unicountvals_2)))
	for f2_it in range(len(fvec)-1):
	    NBvec=NegBinPar(mean_cell_counts_2[f2_it],var_cell_counts_2[f2_it],m2vec)
	    for n2_it in n2itvec:
		Pn2_f[f2_it,n2_it]=np.dot(NBvec[m2_low[n2_it]:m2_high[n2_it]+1],Poisvec2[m2_low[n2_it]:m2_high[n2_it]+1,n2_it]) 
		    
	Pn1n2_s=np.zeros((len(unicountvals_1),len(unicountvals_2))) 
	for n2_it in n2itvec: 
	    tmp1=rhofvec*Pn2_f[:,n2_it]/2.0
	    for n1_it in n1itvec:
		integ=tmp1*Pn1_f[:,n1_it]
		Pn1n2_s[n1_it,n2_it] = np.dot(d,(integ[1:] + integ[:-1]))
	Pn1n2_s/=1.-Pn1n2_s[0,0]
	Pn1n2_s[0,0]=0.
	clonesum=np.sum(countpaircounts_d)
	return -np.dot(countpaircounts_d/float(clonesum),np.where(Pn1n2_s[indn1,indn2]>0,np.log(Pn1n2_s[indn1,indn2]),0))

    else: #s=0 (null model)        
	#print('running Null Model, ')
	Poisvec2=PoisPar(m2vec*r_c2,unicountvals_2) #overwrite. some duplication computation here, but fast enough so leave it.  
	mean_cell_counts_2=m_total*fvec[:-1]
	var_cell_counts_2=mean_cell_counts_2+beta_mv*np.power(mean_cell_counts_2,alpha_mv)
	Pn2_f=np.zeros((len(fvec)-1,len(unicountvals_2)))
	for f2_it in range(len(fvec)-1):
	    NBvec=NegBinPar(mean_cell_counts_2[f2_it],var_cell_counts_2[f2_it],m2vec)
	    for n2_it in n2itvec:
		Pn2_f[f2_it,n2_it]=np.dot(NBvec[m2_low[n2_it]:m2_high[n2_it]+1],Poisvec2[m2_low[n2_it]:m2_high[n2_it]+1,n2_it]) 
	    
	Pn1n2_s=np.zeros((len(unicountvals_1),len(unicountvals_2))) 
	for n2_it,n2 in enumerate(unicountvals_2): 
	    tmp1=rhofvec*Pn2_f[:,n2_it]
	    for n1_it in n1itvec:
		integ=tmp1*Pn1_f[:,n1_it]/ 2.0
		Pn1n2_s[n1_it,n2_it] = np.dot(d,(integ[1:] + integ[-1]))
	Pn1n2_s/=1.-Pn1n2_s[0,0]
	Pn1n2_s[0,0]=0.
	return Pn1n2_s,unicountvals_1,unicountvals_2,Pn1_f,Pn2_f,fvec
      
def callbackF(Xi):
    print('{0: 3.6f}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f} '.format(Xi[0], Xi[1], Xi[2], Xi[3])+'\n')
    
def save_table(outpath, print_expanded, pthresh, svec, Ps,Pn1n2_s, Pn0n0_s,  subset, unicountvals_1_d, unicountvals_2_d,indn1_d,indn2_d):
    Psn1n2_ps=Pn1n2_s*Ps[:,np.newaxis,np.newaxis] 
    
    #compute marginal likelihood (neglect renormalization , since it cancels in conditional below) 
    Pn1n2_ps=np.sum(Psn1n2_ps,0)

    Ps_n1n2ps=Pn1n2_s*Ps[:,np.newaxis,np.newaxis]/Pn1n2_ps[np.newaxis,:,:]
    #compute cdf to get p-value to threshold on to reduce output size
    cdfPs_n1n2ps=np.cumsum(Ps_n1n2ps,0)
    
    def dummy(row,cdfPs_n1n2ps,unicountvals_1_d,unicountvals_2_d):
	return cdfPs_n1n2ps[np.argmin(np.fabs(svec)),row['Clone_count_1']==unicountvals_1_d,row['Clone_count_2']==unicountvals_2_d][0]
    dummy_part=partial(dummy,cdfPs_n1n2ps=cdfPs_n1n2ps,unicountvals_1_d=unicountvals_1_d,unicountvals_2_d=unicountvals_2_d)
    
    cdflabel=r'$1-P(s>0)$'
    subset[cdflabel]=subset.apply(dummy_part, axis=1)
    subset=subset[subset[cdflabel]<pthresh].reset_index(drop=True)
    
    data_pairs_ind_1=np.zeros((len(subset),),dtype=int)
    data_pairs_ind_2=np.zeros((len(subset),),dtype=int)
    for it in range(len(subset)):
	data_pairs_ind_1[it]=np.where(int(subset.iloc[it].Clone_count_1)==unicountvals_1_d)[0]
	data_pairs_ind_2[it]=np.where(int(subset.iloc[it].Clone_count_2)==unicountvals_2_d)[0]   
    #go from pair to index in unique list
    Ps_n1n2ps_datpairs=Ps_n1n2ps[:,data_pairs_ind_1,data_pairs_ind_2]
    
    
    mean_est=np.zeros((len(subset),))
    max_est= np.zeros((len(subset),))
    slowvec= np.zeros((len(subset),))
    smedvec= np.zeros((len(subset),))
    shighvec=np.zeros((len(subset),))
    pval=0.025 #double-sided comparison statistical test
    pvalvec=[pval,0.5,1-pval] #bound in criteria for slow, smed, and shigh, respectively
    for it,column in enumerate(np.transpose(Ps_n1n2ps_datpairs)):
        mean_est[it]=np.sum(svec*column)
        max_est[it]=svec[np.argmax(column)]
        forwardcmf=np.cumsum(column)
        backwardcmf=np.cumsum(column[::-1])[::-1]
        inds=np.where((forwardcmf[:-1]<pvalvec[0]) & (forwardcmf[1:]>=pvalvec[0]))[0]
        slowvec[it]=np.mean(svec[inds+np.ones((len(inds),),dtype=int)])  #use mean in case there are two values
        inds=np.where((forwardcmf>=pvalvec[1]) & (backwardcmf>=pvalvec[1]))[0]
        smedvec[it]=np.mean(svec[inds])
        inds=np.where((forwardcmf[:-1]<pvalvec[2]) & (forwardcmf[1:]>=pvalvec[2]))[0]
        shighvec[it]=np.mean(svec[inds+np.ones((len(inds),),dtype=int)])
    
    colnames=(r'$\bar{s}$',r'$s_{max}$',r'$s_{3,high}$',r'$s_{2,med}$',r'$s_{1,low}$')
    for it,coldata in enumerate((mean_est,max_est,shighvec,smedvec,slowvec)):
        subset.insert(0,colnames[it],coldata)
    oldcolnames=( 'AACDR3',  'ntCDR3', 'Clone_count_1', 'Clone_count_2', 'Clone_fraction_1', 'Clone_fraction_2')
    newcolnames=('CDR3_AA', 'CDR3_nt',        r'$n_1$',        r'$n_2$',           r'$f_1$',           r'$f_2$')
    subset=subset.rename(columns=dict(zip(oldcolnames, newcolnames)))
    print("writing to: "+outpath)
    if print_expanded:
        subset=subset.sort_values(by=cdflabel,ascending=True)
        strout='expanded'
    else:
        subset=subset.sort_values(by=cdflabel,ascending=False)
        strout='contracted'
    subset.to_csv(outpath+'top_'+strout+'.csv',sep='\t',index=False)

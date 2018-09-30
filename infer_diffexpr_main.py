import numpy as np
import time
import sys
from infer_diffexpr_lib import import_data, get_Pn1n2_s, get_sparserep, save_table, get_rhof, get_Ps,callbackF
from functools import partial
from scipy.optimize import minimize
import os
import ctypes
import pandas as pd

#control python multiprocessing:
mkl_rt = ctypes.CDLL('libmkl_rt.so')
num_threads=8 #can set to number of available cores
mkl_set_num_threads = mkl_rt.MKL_Set_Num_Threads(num_threads)
print(str(num_threads)+' cores available')
  
def main(null_pair_1,null_pair_2,test_pair_1,test_pair_2,rootpath):
  
    parvernull = 'v1_manu_version'
    parvertest = 'v1_manu_version'
    
######################################Preprocessing###########################################3

    repstrvec = ('1', '2')

    # filter counts (default no filtering)
    mincount = 0
    maxcount = np.Inf
    
    # script paras
    nfbins=500	#nuber of frequency bins
    smax = 25.0  #maximum absolute logfold change value
    s_step =0.1 #logfold change step size

    nullparatol=1e-4	#tolerance in null model parameter learning
    nullmaxiter=300	#maximum number of iterations allowed (never reached)
#######################################0

    # input/output paras
    path = rootpath + 'outdata/'
    
    # Start Computations
    starttime = time.time()
    
    null_pair_1=null_pair_1.split('.')[0]
    null_pair_2=null_pair_2.split('.')[0]
    test_pair_1=test_pair_1.split('.')[0]
    test_pair_2=test_pair_2.split('.')[0]
    #learn null paras on specified null pair, then load test pair data
    for it,dataset_pair in enumerate(((null_pair_1,null_pair_2),(test_pair_1,test_pair_2))):
      
        #read in data with heterogeneous labelling
	headerline=0
	colnames = [u'Clone fraction',u'Clone count',u'N. Seq. CDR3',u'AA. Seq. CDR3'] 
	print(dataset_pair[0].split('_'))
	donor1,day1str,rep1,_=dataset_pair[0].split('_')
	donor2,day2,rep2,_=dataset_pair[1].split('_')
	assert donor1==donor2, 'trying to compare data from different donors!'
	donorstr=donor1

	datarootpath='Yellow_fever/' 
	
	#input path    
	datasetstr=dataset_pair[0]+'_'+dataset_pair[1] 
	
	loadnull=False
	if (not loadnull and it==0) or it==1:
	  
	    if it==0:
	      	#set output path
		datasetstr_null=datasetstr
		parver=parvernull
		print("running null pair: "+datasetstr)
	    else:
	        parver=parvertest
		parver=datasetstr_null+"_"+parver
		print("running test pair: "+datasetstr)

	    runstr = 'min' + str(mincount) + '_max' + str(maxcount) + '_' + parver
	    outpath = path + dataset_pair[0] + '_' + dataset_pair[1] + '/' + runstr + '/'
	    if not os.path.exists(outpath):
		os.makedirs(outpath)
		
	    #write shelloutput to file
	    outtxtname='_'.join((null_pair_1,null_pair_2,test_pair_1,test_pair_2,'outreport.txt'))
	    outputtxtfile=open(outtxtname, 'w')
	    outputtxtfile.write('outputting to ' + outpath+'\n')
	    if it==0:
		outputtxtfile.write("running null pair: "+datasetstr+'\n')
	    else:
		outputtxtfile.write("running test pair: "+datasetstr+'\n')

	    # import and structure data into a dataframe:
	    Nclones_samp,subset=import_data(datarootpath,dataset_pair[0]+'.txt',dataset_pair[1]+'.txt',mincount,maxcount,colnames,headerline)
	    
	    #transform to sparse representation
	    indn1_d,indn2_d,countpaircounts_d,unicountvals_1_d,unicountvals_2_d,NreadsI_d,NreadsII_d=get_sparserep(subset.loc[:,['Clone_count_1','Clone_count_2']])       
	    Nsamp=np.sum(countpaircounts_d)
	    np.save(outpath+"NreadsI_d.npy",NreadsI_d)
	    np.save(outpath+"NreadsII_d.npy",NreadsII_d)
	    np.save(outpath+"indn1_d.npy",indn1_d)
	    np.save(outpath+"indn2_d.npy",indn2_d)
	    np.save(outpath+"countpaircounts_d.npy",countpaircounts_d)
	    np.save(outpath + 'unicountvals_1_d.npy', unicountvals_1_d)
	    np.save(outpath + 'unicountvals_2_d.npy', unicountvals_2_d)
	    
	    #set sample size adjustment factor
	    repfac=NreadsII_d/NreadsI_d
	    outputtxtfile.write('N_II/N_I='+str(NreadsII_d/NreadsI_d))
	    np.save(outpath+"repfac_prop.npy",repfac)

	    if it==0:
		outputtxtfile.write('learn null:')
		st = time.time()
		partialobjfunc=partial(get_Pn1n2_s,svec=-1,unicountvals_1=unicountvals_1_d, unicountvals_2=unicountvals_2_d, NreadsI=NreadsI_d, NreadsII=NreadsII_d, nfbins=nfbins,repfac=repfac,indn1=indn1_d ,indn2=indn2_d,countpaircounts_d=countpaircounts_d)
		#initial values for null model learning (copied near optimal ones here to speed up):
		donorstrvec=('S1','P2',  'Q2', 'Q1', 'S2','P1')
		defaultnullparasvec=np.asarray( [
		[-2.05 , np.log10(1.4),    1.15,  7.0],  
		[-2.094,         0.334,  1.0517,  7.0],  
		[-2.05 , np.log10(1.7),     1.1,  7.0],  
		[-2.05 , np.log10(0.5),     1.3,  7.0],  
		[-2.05 ,          0.41,   1.107,  6.5],  
		[-2.18 , np.log10(2.4),     1.3,  7.0]  
		])
		dind=[i for i, s in enumerate(donorstrvec) if donorstr in s][0]
		initparas = defaultnullparasvec[dind,:] 
		outstruct = minimize(partialobjfunc, initparas, method='nelder-mead', callback=callbackF, options={'xtol': nullparatol,  'maxiter':nullmaxiter})
		for key,value in outstruct.items():
		    outputtxtfile.write(key+':'+str(value)+'\n')
		if not outstruct.success:
		    print('null learning failed!')
		optparas=outstruct.x
		np.save(outpath + 'optparas', optparas)
		np.save(outpath + 'success', outstruct.success)

		paras=optparas
		et = time.time()
		outputtxtfile.write("elapsed " + str(np.round(et - st))+'\n')
		
		np.save(outpath + 'paras', paras) #null paras to use from here on
	else:
	    datasetstr_null=datasetstr
	    foutname1 = prepath + day1str + '_F' + rep1str 
	    foutname2 = prepath + day2str + '_F' + rep2str
	    runstr = 'min' + str(mincount) + '_max' + str(maxcount) + '_' + parvernull
	    outpath = path + dataset_pair[0] + '_' + dataset_pair[1] + '/' + runstr + '/'
	    paras=	np.load(outpath+'optparas.npy') #
	    print(paras)
    ###############################diffexpr learning
    
    rhofvec,fvec = get_rhof(paras[0],nfbins) #is run again inside get_Pn1n2_s but need fvec here to define svec
    logf_step=np.log(fvec[1]) - np.log(fvec[0]) #use natural log here since f2 increments in exp-based increments.
    f2s_step=int(round(s_step/logf_step)) #rounded number of f-steps in a single s-step
    s_step_old=s_step
    s_step=float(f2s_step)*logf_step
    smax=s_step*(smax/s_step_old)
    outputtxtfile.write("f2s_step:"+str(f2s_step)+"smax:"+str(smax)+", ds:"+str(s_step)+'\n')
    np.save(outpath + 'smax', smax)
    np.save(outpath + 's_step', s_step)
    svec=s_step*np.arange(0,int(round(smax/s_step)+1))
    svec=np.append(-svec[1:][::-1],svec)
    np.save(outpath + 'svec', svec)
    
    #compute conditional
    outputtxtfile.write('calc Pn1n2_s: ')
    st = time.time()
    if os.path.exists(outpath+'Pn1n2_s_d.npy'):
	Pn1n2_s=np.load(outpath+'Pn1n2_s_d.npy')
	Pn0n0_s=np.load(outpath+'Pn0n0.npy')
    else:
	Pn1n2_s, unicountvals_1_d, unicountvals_2_d, Pn1_f, fvec, Pn2_s, Pn0n0_s,svec = get_Pn1n2_s(paras, svec, unicountvals_1_d, unicountvals_2_d,  NreadsI_d, NreadsII_d, nfbins,repfac,indn1=indn1_d,indn2=indn2_d,f2s_step=f2s_step)
	np.save(outpath + 'Pn1n2_s_d', Pn1n2_s)
	np.save(outpath + 'Pn0n0',Pn0n0_s)

    et = time.time()
    outputtxtfile.write("elapsed " + str(np.round(et - st))+'\n')
    
    #######learnSurface:
    outputtxtfile.write('calc surface: \n')
    st = time.time()
    
    #define grid search parameters  
    npoints=20
    nsbarpoints=npoints
    sbarvec=np.linspace(0.01,5,nsbarpoints)
    nalppoints=npoints
    alpvec=np.logspace(-3,np.log10(0.99),nalppoints)

    LSurface =np.zeros((len(sbarvec),len(alpvec)))
    for sit,sbar in enumerate(sbarvec):
	for ait,alp in enumerate(alpvec):
	    Ps=get_Ps(alp,sbar,smax,s_step)
	    Pn0n0=np.sum(Pn0n0_s*Ps)
	    Pn1n2_ps=np.sum(Pn1n2_s*Ps[:,np.newaxis,np.newaxis],0)
	    Pn1n2_ps/=1-Pn0n0
	    LSurface[sit,ait]=np.dot(countpaircounts_d/float(Nsamp),np.where(Pn1n2_ps[indn1_d,indn2_d]>0,np.log(Pn1n2_ps[indn1_d,indn2_d]),0))

    maxinds=np.unravel_index(np.argmax(LSurface),np.shape(LSurface))
    optsbar=sbarvec[maxinds[0]]
    optalp=alpvec[maxinds[1]]
    Psopt=get_Ps(optalp,optsbar,smax,s_step)

    np.save(outpath + 'sbaropt', optsbar)
    np.save(outpath + 'alpopt', optalp)
    np.save(outpath + 'LSurface', LSurface)
    np.save(outpath + 'sbarvec', sbarvec)
    np.save(outpath + 'alpvec', alpvec)
    np.save(outpath + 'Psopt', Psopt)
    outputtxtfile.write("optalp="+str(optalp)+" ("+str(alpvec[0])+","+str(alpvec[-1])+"),optsbar="+str(optsbar)+", ("+str(sbarvec[0])+","+str(sbarvec[-1])+") \n")
    
    et = time.time()
    outputtxtfile.write("elapsed " + str(np.round(et - st))+'\n')

    st=time.time()
    outputtxtfile.write('write table: ')
    optsbar=np.load(outpath + 'sbaropt.npy')
    optalp=np.load(outpath + 'alpopt.npy')
    svec=np.load(outpath + 'svec.npy')
    Pn1n2_s=np.load(outpath + 'Pn1n2_s_d.npy')
    Psopt=np.load(outpath + 'Psopt.npy')
    
    pval_expanded=True #which end of the rank list to pull out. else: most contracted
    pval_threshold=0.1 #output all clones with pval below this threshold
    save_table(outpath+datasetstr+"table",pval_expanded,pval_threshold,svec, Psopt, Pn1n2_s, Pn0n0_s,subset,unicountvals_1_d,unicountvals_2_d,indn1_d,indn2_d)
    et=time.time()
    outputtxtfile.write(" elapsed " + str(np.round(et - st))+'\n')

    # end computations
    endtime = time.time()
    outputtxtfile.write('program elapsed:' + str(endtime - starttime))
    outputtxtfile.close()
    
if __name__ == "__main__": 
    rootpath=''   
    assert len(sys.argv)==5, 'incorrect number of input arguments!'
    inputnull_1=sys.argv[1]
    inputnull_2=sys.argv[2]
    inputtest_1=sys.argv[3]
    inputtest_2=sys.argv[4]
    main(inputnull_1,inputnull_2,inputtest_1,inputtest_2,rootpath)


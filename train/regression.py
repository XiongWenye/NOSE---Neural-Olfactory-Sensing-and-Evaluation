import sys
import warnings
warnings.filterwarnings('ignore')
# !{sys.executable} -m pip i\nstall seaborn

# parent_dir = "/Midgard/home/farzantn/phd/Olfaction/MoLFormer_N2024"
parent_dir = "/public/home/CS182/xiongwy2023-cs182/MoLFormer_N2024"
sys.path.append(parent_dir)

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate,train_test_split
import ast
from sklearn.metrics import roc_auc_score, mean_squared_error
import scipy

# base_path = '/local_storage/datasets/farzaneh/alignment_olfaction_datasets/'
base_path = 'datasets/'
from utils.util_alignment import set_seeds,grand_average,average_over_subject,post_process_results_df
from utils.prepare_datasets import prepare_dataset,select_features
from utils.helper_methods import custom_linear_regression,pipeline_regression,metrics_per_descritor
from constants import *
seed= 2024
set_seeds(seed)
times=30
n_components=20

def train_and_eval(data_groupbyCID,times,n_components=None):
    mserrorrs_corssvalidated = []
    correlations_corssvalidated = []
    predicteds = []
    y_tests = []
    runs = []
    CIDs = []
    
    X=np.asarray(data_groupbyCID.embeddings.values.tolist())
    
    y=np.asarray(data_groupbyCID.y.values.tolist())
    # varss=[]
    for i in range(times):
        X_train, X_test, y_train, y_test,CID_train, CID_test = train_test_split(X, y,data_groupbyCID.CID, test_size=0.2, random_state=seed+i) 
        linreg,X_test,var = pipeline_regression(X_train,y_train,X_test,custom_linear_regression,seed,n_components=n_components)
        
        
        predicted, mseerrors, correlations=metrics_per_descritor(X_test,y_test,linreg)
        mserrorrs_corssvalidated.append(mseerrors)
        correlations_corssvalidated.append(correlations)
        predicteds.extend(predicted)
        y_tests.extend(y_test)
        runs.extend([i]*len(y_test))
        CIDs.extend(CID_test)
        
        
    return CIDs,predicteds,y_tests,runs,mserrorrs_corssvalidated, correlations_corssvalidated

def min_max_extraction(data_groupbyCID,times,y_i=None):
    min_max_dfs = []
    X=np.asarray(data_groupbyCID.embeddings.values.tolist())
    if y_i is not None:
        y=np.asarray(data_groupbyCID.y.values.tolist())[:,y_i].reshape(-1,1)
    else:
       y=np.asarray(data_groupbyCID.y.values.tolist())
    for i in range(times):
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed+i)  
        # print(X_train.shape,"x shape")
        # print(y_train.shape,"y shape")
        df = pd.DataFrame(y_test)

        # Step 3: Extract the min and max values for each column
        min_values = df.min()
        max_values = df.max()

       # Create DataFrames for min and max values with an additional column for the label
        min_df = pd.DataFrame(min_values).T
        min_df['Type'] = 'Min'
    
        max_df = pd.DataFrame(max_values).T
        max_df['Type'] = 'Max'
    
        # Concatenate the min and max DataFrames
        min_max_df = pd.concat([min_df, max_df])
        min_max_df['Dataset'] = i
    
        # Append the concatenated DataFrame to the lis
    
        # Append the min_max_df to the list
        min_max_dfs.append(min_max_df)
    
    final_df = pd.concat(min_max_dfs)   
        # Step 4: Create a new DataFrame with the min and max values per column
    final_df.set_index(['Dataset', 'Type'], inplace=True)
    return final_df
    # min_max_df = pd.DataFrame([min_values, max_values], index=['Min', 'Max'])
        

def pipeline(model_name,input_file,input_file_alva=None,times=30,n_components=None,ds="keller",count=False):
    df_predictions,df_df_mse, df_df_cor = None,None,None
    
    # input_file_keller = base_path+'openpom/data/curated_datasets/embeddings/molformer/keller_molformer_embeddings_13_Apr17.csv'
    df=pd.read_csv(input_file)
    df=prepare_dataset(df,'embeddings','y')
    df_groupbyCID=grand_average(df,ds)
    df_groupbyCIDSubject=average_over_subject(df,ds)
    
    

    if input_file_alva is not None:
        
        df_alva = select_features(input_file_alva)
        df_alva = df_alva.drop_duplicates(subset=['CID'])
        del df_groupbyCID['embeddings']
        df_groupbyCID= pd.merge(df_alva,df_groupbyCID,on="CID")
    
        
    
    if count:
        min_max_df=min_max_extraction(df_groupbyCID,times)
        return min_max_df
    else:
        CIDs, predicteds, y_tests,runs, mserrorrs_df_corssvalidated, correlations_df_corssvalidated=train_and_eval(df_groupbyCID,times=times,n_components=n_components)
   
    mserrorrs_corssvalidated_df,statistics_correlations_corssvalidated_df,pvalues_correlations_corssvalidated_df=post_process_results_df(mserrorrs_df_corssvalidated, correlations_df_corssvalidated)
    df_df_mse= pd.DataFrame(mserrorrs_corssvalidated_df)
    # df_df_mse = df_df_mse.T
    df_df_mse['model'] = model_name
    df_df_cor= pd.DataFrame(statistics_correlations_corssvalidated_df)
    df_df_cor['model'] = model_name
    print(np.asarray(predicteds).shape,np.asarray(y_tests).shape, np.asarray(runs).shape, np.asarray(CIDs).shape)

    # I want to make a dataframe with the predicted values, the true values and the run number for each prediction, (192, 22) (192, 22) (192,) should be converted to (196, 22+22+1), 
    df_predictions = pd.DataFrame(np.concatenate([np.asarray(CIDs).reshape(-1,1),np.asarray(predicteds),np.asarray(y_tests),np.asarray(runs).reshape(-1,1)],axis=1))
    df_predictions['model'] = model_name
    #and add a prefix to the columns to indicate the predicted vs true values
    tasks_length = len(sagar_tasks) if ds.startswith('sagar') else len(keller_tasks)
    df_predictions.columns = ['CID']+[str(i)+'_predicted' for i in range(tasks_length)]+[str(i)+'_true' for i in range(tasks_length,int(tasks_length*2))]+['run']+['model']
    
    
    return df_predictions,df_df_mse, df_df_cor

def compute_correlation(times,n_components,input_file_molformer,input_file_pom,input_file_alva,input_file_molformerfinetuned,ds="keller"):
    df_keller_cor_pom, df_keller_mse_pom, df_keller_cor_alva, df_keller_mse_alva, df_predictions_pom,df_predictions_alva = None,None,None,None,None,None
    
    print("alva")
    df_predictions_alva,df_keller_mse_alva, df_keller_cor_alva = pipeline('alva',input_file_pom,input_file_alva,times=times,n_components=None,ds=ds)
    # 
    corrs_molformer=[]
    corrs_molformerfinetuned = [] 
    # 
    mses_molformer=[]
    mses_molformerfinetuned=[]
    # 
    
    df_predictions_molformers=[]
    df_predictions_molformerfinetuneds=[]

    return corrs_molformer,mses_molformer,df_keller_cor_pom,df_keller_mse_pom,df_keller_cor_alva,df_keller_mse_alva,corrs_molformerfinetuned,mses_molformerfinetuned,df_predictions_molformers,df_predictions_pom,df_predictions_alva,df_predictions_molformerfinetuneds

def count_df_x_keller(times , ds="keller"):

    # for i in [0,13]:
    if ds=="keller":
        input_file_keller_molformer = base_path+'embeddings/molformer/keller_molformer_embeddings_'+str(13)+'.csv'
        min_max_df = pipeline('molformer',input_file_keller_molformer,times=times,n_components=n_components,count=True)
    return min_max_df

def post_process_tocsv(corrs,tasks,title):
    corrs[0]["layer"]=0
    corrss = corrs[0]
    for i in range(1,13):
        corrs[i]["layer"] = i
        corrss  = pd.concat([corrss, corrs[i]])
        print("i", i )
    del corrss['model']
    print(corrss.columns.values.tolist(),"columns")
    corrss.columns = tasks+["layer"]    
    corrss['model']=title
    return corrss

def save_data(ds,df_cor_pom,df_cor_alva,df_mse_pom,df_mse_alva,corrs_molfomer,mses_molformer, corrs_molfomerfinetuned,mses_molfomerfinetuned):
    if ds=="keller":
        tasks= keller_tasks
    
    df_cor_pom.columns = tasks+["model"]
    df_cor_pom.to_csv('df_'+ds+'_cor_pom.csv', index=False)   
    df_mse_pom.columns  = tasks+["model"]
    df_mse_pom.to_csv('df_'+ds+'_mse_pom.csv', index=False)  
    # 
    df_cor_alva.columns = tasks+["model"]
    df_cor_alva.to_csv('df_'+ds+'_cor_alvanotnan.csv', index=False)

    df_mse_alva.columns = tasks+["model"]
    df_mse_alva.to_csv('df_'+ds+'_mse_alvanotnan.csv', index=False)
     
    corrs_molfomer_df = post_process_tocsv(corrs_molfomer,tasks)
    corrs_molfomer_df.to_csv('df_'+ds+'_corrs_molfomer.csv', index=False)    
    mses_molformer_df = post_process_tocsv(mses_molformer,tasks)
    mses_molformer_df.to_csv('df_'+ds+'_mses_molfomer.csv', index=False)   
    # 
    corrs_molfomer_df = post_process_tocsv(corrs_molfomer,tasks)
    corrs_molfomer_df.to_csv('df_'+ds+'_corrs_molfomer.csv', index=False)   
    # 
    mses_molformer_df = post_process_tocsv(mses_molformer,tasks)
    mses_molformer_df.to_csv('df_'+ds+'_mses_molfomer.csv', index=False)


    # 
    # df_cor_alva.columns = tasks+["model"]
    # df_cor_alva.to_csv('df_'+ds+'_cor_mordred.csv', index=False)
    # 
    # df_mse_alva.columns = tasks+["model"]
    # df_mse_alva.to_csv('df_'+ds+'_mse_mordred.csv', index=False)

    corrs_molfomerfinetuned_df = post_process_tocsv(corrs_molfomerfinetuned,tasks,"molformerfinetuned")
    corrs_molfomerfinetuned_df.to_csv('df_'+ds+'_corrs_molfomerfinetuned.csv', index=False)   
    # 
    mses_molfomerfinetuned_df = post_process_tocsv(mses_molfomerfinetuned,tasks,"molformerfinetuned")
    mses_molfomerfinetuned_df.to_csv('df_'+ds+'_mses_molfomerfinetuned.csv', index=False)   

def concat_dfs(df_predictions_molformers,df_predictions_pom,df_predictions_alva):
    df_predictions = pd.concat([df_predictions_molformers[0],df_predictions_molformers[1],df_predictions_molformers[2],df_predictions_molformers[3],df_predictions_molformers[4],df_predictions_molformers[5],df_predictions_molformers[6],df_predictions_molformers[7],df_predictions_molformers[8],df_predictions_molformers[9],df_predictions_molformers[10],df_predictions_molformers[11],df_predictions_molformers[12],df_predictions_pom,df_predictions_alva])
    return df_predictions

input_file_keller_pom = base_path+'embeddings/pom/keller_pom_embeddings_.csv'
input_file_keller_molformer = base_path+'embeddings/molformer/keller_molformer_embeddings_'
input_file_keller_molformerfinetuned = base_path+'embeddings/molformerfinetuned/keller_molformerfinetuned_embeddings_'
corrs_molfomer,mses_molformer,df_keller_cor_pom,df_keller_mse_pom,df_keller_cor_alva,df_keller_mse_alva,df_keller_cor_molformerfinetuned,df_keller_mse_molformerfinetuned,df_predictions_molformers,df_predictions_pom,df_predictions_alva,df_predictions_molformerfinetuned =compute_correlation(times, n_components,input_file_keller_molformer,input_file_keller_pom,input_file_keller_molformerfinetuned,ds="keller2")
# pd.read_csv(input_file_keller_molformerfinetuned+str(0)+'_model_1_Apr17.csv')
save_data("keller",df_keller_cor_pom,df_keller_cor_alva,df_keller_mse_pom,df_keller_mse_alva,corrs_molfomer,mses_molformer,df_keller_cor_molformerfinetuned,df_keller_mse_molformerfinetuned)
min_max_df =count_df_x_keller(times )
min_max_df.to_csv('keller_min_max.csv', index=True)
 

import pandas as pd
from utils.util_alignment import set_seeds
from utils.visualization_helper import *  
import matplotlib.pyplot as plt
from constants import *
plt.rc('font',**{'family':'serif','serif':['Calibri']})
results_path = 'dfs_result/regression/'

def normalize_rmse(df,min_max,j):
    max_values = min_max.max()
    min_values = min_max.min()
    min_max.columns = ['Dataset','Type']+df.columns.values.tolist()[:j]
    # Drop 'Dataset' and 'Type' as they are not numeric columns
    min_values = min_values.drop(['Dataset', 'Type'])
    max_values = max_values.drop(['Dataset', 'Type'])
    for i,col in enumerate(df.columns[:j]):
        df[col] = np.sqrt(df[col]) / (max_values[i] - min_values[i])
    return df

input_file_keller = 'dataset/keller2016_binarized.csv'
df_keller_temp=pd.read_csv(input_file_keller)

df_keller_cor_pom=pd.read_csv(base_path+results_path+"df_keller_cor_pom.csv")
df_keller_mse_pom=pd.read_csv(base_path+results_path+"df_keller_mse_pom.csv")
df_keller_corrs_molfomer=pd.read_csv(base_path+results_path+"df_keller_corrs_molfomer.csv")
df_keller_mses_molfomer=pd.read_csv(base_path+results_path+"df_keller_mses_molfomer.csv")
df_keller_corrs_molfomerfinetuned=pd.read_csv(base_path+results_path+"df_keller_corrs_molfomerfinetuned.csv")
df_keller_mses_molfomerfinetuned=pd.read_csv(base_path+results_path+"df_keller_mses_molfomerfinetuned.csv")

min_max_keller = pd.read_csv(base_path+results_path+"keller_min_max.csv")
df_keller_mse_pom=normalize_rmse(df_keller_mse_pom,min_max_keller,j=-1)
df_keller_mses_molfomer=normalize_rmse(df_keller_mses_molfomer,min_max_keller,j=-2)
df_keller_mses_molfomerfinetuned=normalize_rmse(df_keller_mses_molfomerfinetuned,min_max_keller,j=-2)

trend_learning_molformer =  post_process_dataframe(df_keller_corrs_molfomer,df_keller_mses_molfomer,df_keller_corrs_molfomerfinetuned,df_keller_mses_molfomerfinetuned,df_keller_cor_pom,df_keller_mse_pom,df_keller_cor_alva,df_keller_mse_alva,keller_tasks,"figs/keller_regression_finetune",width=None,linewidth=0)
# trend_learning_molformer['dataset']='keller'
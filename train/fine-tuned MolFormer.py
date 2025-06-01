import os
import sys
parent_dir = "/public/home/CS182/xiongwy2023-cs182/MoLFormer_N2024"
sys.path.append(parent_dir) 

from finetune_multitask.py import MultitaskModel
tokenizer = MolTranBertTokenizer('custom_utils/tokenizer/bert_vocab.txt')
# len(tokenizer.vocab)
def set_lm_frozen(ckpt):
    lm = LightningModule(config, tokenizer.vocab).load_from_checkpoint(ckpt, config=config, vocab=tokenizer.vocab)
    return lm
# 
# def set_lm_finetuned(ckpt):
#     
#     return lm
    
    

  # model = MultitaskModel(margs, tokenizer)
  #   else:
  #       print("# loaded pre-trained model from {args.seed_path}")
  #       model = MultitaskModel(margs, tokenizer).load_from_checkpoint(margs.seed_path, strict=False, config=margs, tokenizer=tokenizer, vocab=len(tokenizer.vocab))
        
ckpt_frozen = 'MoLformer_Pretrained/checkpoints/N-Step-Checkpoint_3_30000.ckpt'
lm_frozen = set_lm_frozen(ckpt=ckpt_frozen)

def extract_molformer_finetuned(ds,input_file,cids,cids_subject,tasks):
    for j in range(29,31):
        print("j",j)
        ckpt = 'checkpoints_gs_lf/curated_GS_LF_merged_4983/'+str(j)+'/models_0000/checkpoint.ckpt'
        tokenizer = MolTranBertTokenizer('custom_utils/tokenizer/bert_vocab.txt')
        print(len(tokenizer.vocab))
        # lm  = set_lm_finetuned(ckpt)
        config.num_tasks = len(gs_lf_tasks)
        config.dims = [768,768,768,1]
        lm = MultitaskModel(config, tokenizer).load_from_checkpoint(ckpt, config=config,tokenizer=tokenizer, vocab=len(tokenizer.vocab))
        
        X,y,X_layers,y_layers=extract_molformer_representations(lm, tokenizer, tasks, input_file, smiles_field)
        print("X",X.shape)
        # print("y",y.shape)
        
        
        # return X,y,X_layers,y_layers
        df_embeddings = convert_todf_molformer(X,cids,cids_subject,y)
        df_embeddings.to_csv(ds+'_molformerfinetuned_embeddings_13'+"_model_"+str(j)+'.csv',     index=False)
        for i,X in enumerate(X_layers):
            print("i",i)
            df_embeddings = convert_todf_molformer(X_layers[i],cids,cids_subject,y_layers[i])
            df_embeddings.to_csv(ds+'_molformerfinetuned_embeddings_'+str(i)+"_model_"+str(j)+'.csv', index=False)

def convert_todf_molformer(embeddings_dataset,cids,subjects=None,y=None):
    embeddings_dataset_copy = pd.DataFrame(embeddings_dataset)
    embeddings_dataset = pd.DataFrame()
    embeddings_dataset['embeddings'] = embeddings_dataset_copy.loc[:, 0:768].values.tolist()
    embeddings_dataset['CID'] = cids
    if subjects is not None:
        embeddings_dataset['subject'] = subjects
        
    if y is not None:
        y_dataset = pd.DataFrame(y)
        y_dataset['y'] = y_dataset.loc[:, 0:768].values.tolist()
    
        df = pd.concat([embeddings_dataset, y_dataset], axis=1)
        return df
    else:
        return embeddings_dataset
    
input_file_gslf = 'dataset/curated_GS_LF_merged_4983.csv' 
smiles_field = 'nonStereoSMILES'
df_gslf_temp=pd.read_csv(input_file_gslf)
gslf_tasks=df_gslf_temp.columns.to_list()[2:]
cids_gslf= df_gslf_temp.index.values.tolist()
# X_gslf,y_gslf,X_layers_gslf,y_layers_gslf=extract_molformer_representations(lm, tokenizer, gslf_tasks, input_file_gslf, smiles_field)

X_gslf, y_gslf, X_layers_gslf, y_layers_gslf = extract_molformer_representations(lm_frozen, tokenizer, gslf_tasks,
                                                                                 input_file_gslf, smiles_field)
df_embeddings_gslf = convert_todf_molformer(X_gslf,cids_gslf,None,y_gslf)
df_embeddings_gslf.to_csv('gslf_molformer_embeddings_13.csv', index=False)
for i,X in enumerate(X_layers_gslf[:2]):
    # print("i",i)
    df_embeddings_gslf = convert_todf_molformer(X_layers_gslf[i],cids_gslf,None,y_layers_gslf[i])
    df_embeddings_gslf.to_csv('gslf_molformer_embeddings_'+str(i)+'.csv', index=False)

extract_molformer_finetuned('gslf',input_file_gslf,cids_gslf,None,gslf_tasks)


input_file_keller = 'dataset/keller2016_binarized.csv' 
smiles_field = 'nonStereoSMILES'
df_keller_temp=pd.read_csv(input_file_keller)
keller_tasks= df_keller_temp.columns.to_list()[5:]
cids_keller= df_keller_temp['CID'].values.tolist()
cids_subject_keller= df_keller_temp['Subject'].values.tolist()

X_keller,y_keller,X_layers_keller,y_layers_keller=extract_molformer_representations(lm_frozen, tokenizer, keller_tasks, input_file_keller, smiles_field)
df_embeddings_keller = convert_todf_molformer(X_keller,cids_keller,cids_subject_keller,y_keller)
df_embeddings_keller.to_csv('keller_molformer_embeddings_13.csv', index=False)
for i,X in enumerate(X_layers_keller):
    print("i",i)
    df_embeddings_keller = convert_todf_molformer(X_layers_keller[i],cids_keller,cids_subject_keller,y_layers_keller[i])
    df_embeddings_keller.to_csv('keller_molformer_embeddings_'+str(i)+'.csv', index=False)

extract_molformer_finetuned('keller',input_file_keller,cids_keller,cids_subject_keller,keller_tasks)
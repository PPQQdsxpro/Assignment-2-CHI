"""Prepare disjoint data and evaluate a fixed TF-IDF logistic regression pipeline."""
from pathlib import Path
import csv,json,re,hashlib,random,sys,platform,importlib.metadata
import pandas as pd
import jieba
import joblib
from text_features import tokens
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.dummy import DummyClassifier
ROOT=Path(__file__).resolve().parents[1]
SEED=42


def key(text): return re.sub(r'\s+','',str(text))
def save_json(name,value):
    (ROOT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    rng=random.Random(SEED)
    frames={}
    for split in ['train','dev','test']:
        df=pd.read_csv(ROOT/f'data/raw/{split}.csv',keep_default_na=False)
        df=df.rename(columns={df.columns[0]:'source_id','TEXT':'text'})
        df['source_file']=split+'.csv';df['source_row']=range(2,len(df)+2);df['key']=df.text.map(key)
        frames[split]=df
    allrows=pd.concat(frames.values(),ignore_index=True)
    conflicts=set(allrows.groupby('key').label.nunique().loc[lambda x:x>1].index)
    blind=json.loads((ROOT/'review/reference_labels.json').read_text(encoding='utf-8'))
    reviewed={key(r['text']) for r in blind}
    # Reserve the whole official test partition for application, never model selection.
    reserved=set(frames['test']['key'])
    eligible=frames['train'].loc[lambda x:~x.key.isin(conflicts|reserved|reviewed) & x.key.ne('')].drop_duplicates('key')
    selected=pd.concat([eligible[eligible.label==label].sample(n=500,random_state=SEED) for label in [0,1]],ignore_index=True)
    train,test=train_test_split(selected,test_size=.2,stratify=selected.label,random_state=SEED)
    apppool=frames['test'].loc[lambda x:~x.key.isin(conflicts|reviewed)&x.key.ne('')].drop_duplicates('key')
    app=pd.concat([apppool[apppool['fine-grained-label']==i].sample(n=25,random_state=SEED) for i in range(4)],ignore_index=True)
    for a,b in [(train,test),(selected,app)]:assert not(set(a.key)&set(b.key))
    for name,df in [('labeled_data',selected),('train_split',train),('test_split',test),('application_data',app)]:
        df.drop(columns='key').to_csv(ROOT/f'data/{name}.csv',index=False,encoding='utf-8-sig')
    # Vocabulary and IDF learned inside each training fold only.
    model=Pipeline([('tfidf',TfidfVectorizer(tokenizer=tokens,token_pattern=None,lowercase=False,min_df=2,ngram_range=(1,1))),('classifier',LogisticRegression(C=1,max_iter=2000,random_state=SEED))])
    cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=SEED)
    scores=cross_val_score(model,train.text,train.label,cv=cv,scoring='f1_macro')
    model.fit(train.text,train.label)
    (ROOT/'models').mkdir(exist_ok=True)
    joblib.dump(model,ROOT/'models/cold_logistic.joblib')
    pred=model.predict(test.text)
    baseline=DummyClassifier(strategy='most_frequent').fit([[0]]*len(train),train.label)
    metrics={'accuracy':accuracy_score(test.label,pred),'classification_report':classification_report(test.label,pred,output_dict=True,zero_division=0),'confusion_matrix_rows_true_columns_predicted_0_1':confusion_matrix(test.label,pred,labels=[0,1]).tolist(),'training_only_cv_macro_f1':scores.tolist(),'baseline_accuracy':accuracy_score(test.label,baseline.predict([[0]]*len(test)))}
    for name,df in [('test_predictions',test),('application_predictions',app)]:
        result=df.drop(columns='key').copy();result['prediction']=model.predict(df.text);result['p_offensive']=model.predict_proba(df.text)[:,1];result['matches_reference']=result.label==result.prediction
        result.to_csv(ROOT/f'output/{name}.csv',index=False,encoding='utf-8-sig')
    features=model.named_steps['tfidf'].get_feature_names_out();weights=model.named_steps['classifier'].coef_[0]
    pd.DataFrame({'feature':features,'weight':weights}).sort_values('weight',ascending=False).to_csv(ROOT/'output/feature_weights.csv',index=False,encoding='utf-8-sig')
    save_json('output/evaluation.json',metrics)
    save_json('output/run_metadata.json',{'seed':SEED,'model':'TF-IDF unigrams min_df=2 + LogisticRegression C=1 max_iter=2000','stopwords':'none; punctuation removed; negation retained','normalization':'no script conversion; whitespace ignored only for duplicate comparison','conflicting_unique_texts_all_splits':len(conflicts),'reviewed_texts_excluded':len(reviewed),'eligible_train':len(eligible),'model_data':len(selected),'training':len(train),'held_out_test':len(test),'application':len(app),'application_sampling':'25 per fine-grained category; not natural prevalence','unused':'official dev not used','duplicate_check':'exact after whitespace removal; semantic/near duplicates not fully excluded','packages':{p:importlib.metadata.version(p) for p in ['scikit-learn','jieba','pandas','numpy','scipy']},'raw_sha256':{s:hashlib.sha256((ROOT/f'data/raw/{s}.csv').read_bytes()).hexdigest() for s in frames}})
    print(json.dumps({'training':len(train),'test':len(test),'application':len(app),'accuracy':metrics['accuracy'],'macro_f1':metrics['classification_report']['macro avg']['f1-score'],'confusion_matrix':metrics['confusion_matrix_rows_true_columns_predicted_0_1']},ensure_ascii=False))
if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');main()


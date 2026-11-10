from pathlib import Path
import pandas as pd,json,re,html
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
root=Path.cwd();train=pd.read_csv('data/train_split.csv');test=pd.read_csv('data/test_split.csv');app=pd.read_csv('data/application_data.csv')
frames=[train,test,app];alltext=pd.concat(frames).text.astype(str).map(lambda s:re.sub(r'\s+','',s)).tolist()
v=TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=1);x=v.fit_transform(alltext)
records=[];starts=[0,len(train),len(train)+len(test)]
for a,b in [(0,1),(0,2),(1,2)]:
 sim=cosine_similarity(x[starts[a]:starts[a]+len(frames[a])],x[starts[b]:starts[b]+len(frames[b])])
 for i,j in zip(*(sim>=.65).nonzero()):
  ra=frames[a].iloc[i];rb=frames[b].iloc[j];ratio=SequenceMatcher(None,alltext[starts[a]+i],alltext[starts[b]+j],autojunk=False).ratio()
  records.append({'split_a':['train','test','application'][a],'split_b':['train','test','application'][b],'source_a':int(ra.source_row),'source_b':int(rb.source_row),'cosine':float(sim[i,j]),'sequence_ratio':ratio,'text_a':ra.text,'text_b':rb.text})
(root/'output/near_duplicate_audit.json').write_text(json.dumps({'method':'char 3-5gram TF-IDF cosine >=0.65, followed by sequence comparison; heuristic, not exhaustive semantic detection','pairs':records},ensure_ascii=False,indent=2),encoding='utf-8')
# Only training records: no held-out labels are revised using this review.
sample=pd.concat([train[train.label==k].sample(14,random_state=43) for k in [0,1]]).sample(frac=1,random_state=43)
blind=[];parts=['<h1>補充訓練資料抽查：28則</h1><p>只取800則訓練資料，兩類各14則，種子43；原標籤隱藏。每則請填0/1/U、信心及有疑問時的理由。這是訓練資料審核，不是重新判定測試答案。</p>']
for i,(_,r) in enumerate(sample.iterrows(),1):
 rid=f'B{i:02d}';blind.append({'review_id':rid,'source_row':int(r.source_row),'text':r.text,'reference':int(r.label),'user_label':None})
 parts.append(f'<article><h2>{rid}</h2><p>{html.escape(r.text)}</p><p>標籤：＿＿　信心：＿＿　理由／疑問：＿＿</p></article>')
(root/'review/training_audit_28.json').write_text(json.dumps(blind,ensure_ascii=False,indent=2),encoding='utf-8')
(root/'output/training_audit.html').write_text('<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>訓練資料抽查</title><style>body{max-width:950px;margin:30px auto;padding:20px;font:18px/1.8 "Microsoft JhengHei",sans-serif;background:#f7f3ed}article{background:white;padding:20px;margin:18px 0;border:1px solid #ccc}</style>'+''.join(parts)+'</html>',encoding='utf-8')
print(json.dumps(records,ensure_ascii=False,indent=2));print('28 review cases prepared')

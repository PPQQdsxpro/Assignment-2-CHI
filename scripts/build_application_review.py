from pathlib import Path
import sys,json,html
sys.stdout.reconfigure(encoding='utf-8');sys.path.insert(0,str(Path('scripts').resolve()))
import pandas as pd,numpy as np,joblib
root=Path.cwd();m=joblib.load('models/cold_logistic.joblib');v=m.named_steps['tfidf'];coef=m.named_steps['classifier'].coef_[0];names=v.get_feature_names_out()
a=pd.read_csv('output/application_predictions.csv');b=a[a['fine-grained-label']==3]
chosen=pd.concat([b[b.prediction==1].nlargest(3,'p_offensive'),b[b.prediction==0].nsmallest(2,'p_offensive'),b.assign(distance=(b.p_offensive-.5).abs()).nsmallest(1,'distance')]).drop_duplicates('source_row')
records=[]
for i,(_,r) in enumerate(chosen.iterrows(),1):
 x=v.transform([r.text]).toarray()[0];indices=np.flatnonzero(x);rank=sorted(indices,key=lambda j:abs(x[j]*coef[j]),reverse=True)[:8]
 records.append({'case_id':f'A{i:02d}','source_row':int(r.source_row),'text':r.text,'reference':int(r.label),'fine_category':3,'prediction':int(r.prediction),'p_offensive':float(r.p_offensive),'contributions':[{'word':str(names[j]),'value':round(float(x[j]*coef[j]),4)} for j in rank],'ignored_words':sorted(set(t for t in v.build_analyzer()(r.text) if t not in v.vocabulary_))})
Path('review/application_cases.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
parts=['<h1>反偏見留言：應用階段細讀</h1><p>25則原標籤為反偏見的留言，11則判為非冒犯、14則判為冒犯。這是每細類25則的目的性抽樣，不代表平台自然分布。以下選3則最高分誤報、2則最低分正確例、1則最接近門檻者，不代表隨機樣本。</p><p>請先讀原文，再展開模型資料。每則填：你判0/1/U；作者反對甚麼、針對誰；有無同時辱罵；同意模型還是參考標籤；證據。</p>']
for r in records:
 parts.append(f'<article><h2>{r["case_id"]}</h2><p>{html.escape(r["text"])}</p><details><summary>模型數值與詞彙</summary><p>原標籤0（反偏見）；預測{r["prediction"]}；冒犯機率{r["p_offensive"]:.3f}。來源：test.csv記錄第{r["source_row"]}行。</p>'+pd.DataFrame(r['contributions']).to_html(index=False)+'<p>未納入詞彙：'+html.escape('、'.join(r['ignored_words']))+'</p></details><p>你的筆記：＿＿＿＿</p></article>')
Path('output/application_review.html').write_text('<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>反偏見案例細讀</title><style>body{max-width:950px;margin:40px auto;padding:0 20px;background:#f6f3ed;color:#243b36;font:18px/1.8 "Microsoft JhengHei",sans-serif}article{background:white;padding:22px;margin:20px 0;border:1px solid #ccd5d0;border-radius:10px}td,th{border:1px solid #ccc;padding:6px}table{border-collapse:collapse}summary{cursor:pointer}</style>'+''.join(parts)+'</html>',encoding='utf-8')
for r in records:print(r['case_id']+': '+r['text'])

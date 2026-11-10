from pathlib import Path
import json,html
import pandas as pd
import numpy as np
import joblib
ROOT=Path(__file__).resolve().parents[1]
def main():
 model=joblib.load(ROOT/'models/cold_logistic.joblib')
 test=pd.read_csv(ROOT/'output/test_predictions.csv');app=pd.read_csv(ROOT/'output/application_predictions.csv')
 app.groupby('fine-grained-label').agg(n=('label','size'),agreement=('matches_reference','mean'),mean_p_offensive=('p_offensive','mean')).to_csv(ROOT/'output/application_group_summary.csv',encoding='utf-8-sig')
 assert np.allclose(model.predict_proba(test.text)[:,1],test.p_offensive)
 metrics=json.loads((ROOT/'output/evaluation.json').read_text());weights=pd.read_csv(ROOT/'output/feature_weights.csv')
 names=model.named_steps['tfidf'].get_feature_names_out();coef=model.named_steps['classifier'].coef_[0]
 errors=test[test.label!=test.prediction]
 # Deterministic cases: strongest false positives/negatives and uncertain errors.
 chosen=pd.concat([errors[errors.prediction==1].nlargest(3,'p_offensive'),errors[errors.prediction==0].nsmallest(3,'p_offensive'),errors.assign(distance=(errors.p_offensive-.5).abs()).nsmallest(2,'distance')]).drop_duplicates(['source_file','source_row'])
 sections=['<h1>中文冒犯性留言分類：結果與細讀</h1><p>模型已保存並重新載入，200筆預測機率與原結果一致。這是固定模型的結果；目前的細讀不再用來調整保留測試集上的分數。</p>','<h2>模型在哪裏？</h2><p>models/cold_logistic.joblib 保存詞彙表、IDF與邏輯迴歸權重，需Python載入；本頁供人閱讀。scripts/run_pipeline.py 是建立模型的流程，scripts/text_features.py 是分詞函數。</p>','<h2>三折交叉驗證</h2><p>只在800筆訓練資料內輪流訓練與驗證，每折重新學習TF-IDF。下表是Macro-F1，不是準確率；200筆保留測試資料不參與。</p>']
 cv=metrics['training_only_cv_macro_f1'];sections.append(pd.DataFrame({'折':['1','2','3','平均'],'Macro-F1':cv+[sum(cv)/3]}).round(4).to_html(index=False))
 sections+=['<h2>保留測試集</h2><p>準確率82%；多數類別基準50%。橫欄為預測，直欄為資料集標籤。</p>',pd.DataFrame(metrics['confusion_matrix_rows_true_columns_predicted_0_1'],index=['實際非冒犯','實際冒犯'],columns=['預測非冒犯','預測冒犯']).to_html(),'<h2>重要特徵</h2><p>正權重推向冒犯，負權重推向非冒犯；權重不是詞語本質。每句的實際影響還要乘該句的TF-IDF。</p>',pd.concat([weights.head(15),weights.tail(15)]).round(4).to_html(index=False)]
 sections.append('<h2>八則誤判工作紙</h2><p>依模型分數選取高信心誤報、漏報和接近門檻的錯誤；屬目的性選例，不代表自然比例。先看留言，再展開數值。請寫：你判0/1/U、關鍵證據、同意模型還是參考標籤、缺少甚麼語境。</p>')
 cases=[]
 for i,(_,r) in enumerate(chosen.iterrows(),1):
  x=model.named_steps['tfidf'].transform([r.text]).toarray()[0];con=x*coef;indices=np.flatnonzero(x);rank=sorted(indices,key=lambda j:abs(con[j]),reverse=True)[:8]
  contrib=[{'word':str(names[j]),'contribution':round(float(con[j]),4)} for j in rank]
  cases.append({'case_id':f'E{i:02d}','source_row':int(r.source_row),'text':r.text,'reference':int(r.label),'prediction':int(r.prediction),'p_offensive':float(r.p_offensive),'top_contributions':contrib})
  sections.append(f'<article><h3>E{i:02d} · 原始train.csv記錄第{r.source_row}行</h3><p>{html.escape(r.text)}</p><details><summary>查看預測與詞語貢獻</summary><p>參考標籤 {r.label}；模型 {r.prediction}；冒犯機率 {r.p_offensive:.3f}</p><p>{html.escape(str(contrib))}</p></details><p>你的判斷：＿＿　理由：＿＿　語境／標籤疑問：＿＿</p></article>')
 sections+=['<h2>獨立應用資料：四種細類</h2><p>每類25則，仍來自COLD，不能宣稱跨平台泛化。以下數值是與原標籤一致程度，不是人工裁定的真理。</p>',app.groupby('fine-grained-label').agg(樣本數=('label','size'),與原標籤一致比例=('matches_reference','mean'),平均冒犯機率=('p_offensive','mean')).round(3).to_html(),'<p>細類：0其他非冒犯；1攻擊個人；2攻擊群體；3反偏見。</p>','<h2>全部36則測試分歧</h2>',errors[['source_row','text','label','prediction','p_offensive']].round(3).to_html(index=False)]
 page='<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>作業二結果與細讀</title><style>body{font:17px/1.8 "Microsoft JhengHei",sans-serif;max-width:1050px;margin:40px auto;padding:0 22px;color:#243b36;background:#f6f3ed}table{border-collapse:collapse;width:100%;background:white;margin:20px 0}td,th{border:1px solid #bccbc5;padding:8px;text-align:left;overflow-wrap:anywhere}article{background:white;border:1px solid #ccd5d0;border-radius:10px;padding:20px;margin:18px 0}h2{margin-top:40px}summary{cursor:pointer;color:#246354}details p{overflow-wrap:anywhere}</style>'+''.join(sections)+'</html>'
 (ROOT/'output/results.html').write_text(page,encoding='utf-8');(ROOT/'review/error_cases.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2),encoding='utf-8')
 print('Reload verified; results.html and 8 error cases created.');print(app.groupby('fine-grained-label').matches_reference.mean().to_string())
if __name__=='__main__':main()

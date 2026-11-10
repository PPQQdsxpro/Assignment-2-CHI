# 作業二：中文冒犯性留言分類

作業二獨立專案：中文留言的冒犯性語言分類。

研究問題：詞袋分類器能否辨識中文留言的冒犯性表達？引用、反偏見與隱含貶抑帶來甚麼判斷困難？

來源：COLD 官方資料集 https://github.com/thu-coai/COLDataset ，原論文 https://aclanthology.org/2022.emnlp-main.796/ 。本次原始 CSV 由使用者提供，已保存檔案雜湊。標籤 0=非冒犯，1=冒犯；這些是資料集原有標籤，不宣稱全部由學生從零標註。

## 流程

從官方 train 去除空白、精確重複、跨來源標籤衝突、所有官方 test 的同文及12則規則開發留言後，每類抽500筆。分層80/20切分為800筆訓練、200筆測試，seed=42。官方 dev 僅用於重複及標籤衝突稽核，不參與建模。官方 test 另按四個細類各抽25筆作應用資料；同來源應用不能證明跨平台泛化。

TF-IDF單詞特徵（min_df=2）與邏輯迴歸（C=1），不刪否定詞。三折交叉驗證僅在訓練部分進行；詞彙表與IDF在各折內學習。固定單一模型，不依保留測試結果調參。近重複及語意改寫尚未全面排除，報告須列為限制。

## 執行

Python 3.12，安裝 jieba、pandas、scikit-learn；實際版本見 output/run_metadata.json。

```text
python scripts/run_pipeline.py
```

## 文件

- data/raw：原始檔案副本（兩份相同test只保留一份）。
- data：標註資料、固定切分及應用資料。
- review：學生盲判、參考標籤與標註分歧，保存規則開發、訓練抽查及測試與應用案例細讀。
- output：評估、逐筆預測、特徵權重及參數紀錄。

學生已完成12則規則試標、28則訓練留言抽查，以及8則測試案例與6則應用案例細讀。AI協助程式、統計核對、案例解釋及依學生筆記整合報告。完整使用聲明見 report.md。報告與PDF已完成；GitHub保存研究材料，Moodle須另提交倉庫網址與PDF。

report.md、report.pdf 為同一份報告；output/report.html 可在瀏覽器閱讀。報告含分類邊界、冒犯與反偏見的區別，以及理解模型和驗證方法時遇到的困難。正文連標題、表格共2,094個漢字，不含參考資料與AI聲明。

重建閱讀頁及報告：

```text
python scripts/build_results.py
python scripts/build_application_review.py
python scripts/render_report.py
```

PDF生成另需 markdown 與 reportlab，並使用 Windows 微軟正黑體 C:/Windows/Fonts/msjh.ttc；其他系統須調整字體路徑。

## 查看模型與結果

models/cold_logistic.joblib 是完整模型（需配合 scripts/text_features.py）。python scripts/build_results.py 會重新載入並核對預測，生成 output/results.html 與 review/error_cases.json。結果頁包含三折交叉驗證、混淆矩陣、特徵權重及八則細讀案例。


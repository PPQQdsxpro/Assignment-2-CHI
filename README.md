# 作業二：辨識冒犯，還是辨識冒犯詞？

研究問題：詞袋分類器能否辨識中文留言的冒犯性表達？引用、反偏見與隱含貶抑等....帶來甚麼判斷困難？

**Python程式集中在 [scripts/](scripts/)，並沒有缺少。** [run_pipeline.py](scripts/run_pipeline.py)包含準備資料、切分、三折交叉驗證、訓練、評估及新文本應用。

## 作業要求與檔案位置

| 要求／用途 | 檔案連結 | 內容 |
|---|---|---|
| 報告兩種格式 | [report.md](report.md)、[report.pdf](report.pdf) | 學生修訂稿加上指定的反偏見困難段落；正文2,305漢字，含標題表格，不含參考資料與AI聲明 |
| 標註建模資料 | [data/labeled_data.csv](data/labeled_data.csv) | 1,000則，每類500則；text為文本，label為0或1 |
| 訓練集 | [data/train_split.csv](data/train_split.csv) | 800則，每類400則 |
| 保留測試集 | [data/test_split.csv](data/test_split.csv) | 200則，每類100則 |
| 另一組分析文本 | [data/application_data.csv](data/application_data.csv) | 100則未參與建模的官方test留言，四細類各25則 |
| 準備、切分、訓練、評估與應用 | [scripts/run_pipeline.py](scripts/run_pipeline.py) | 同一腳本按順序完成全部建模步驟 |
| 分詞程式 | [scripts/text_features.py](scripts/text_features.py) | jieba分詞及純標點移除；載入模型亦需此檔 |
| 測試集逐筆預測 | [output/test_predictions.csv](output/test_predictions.csv) | 原文、參考標籤、prediction、p_offensive及是否一致 |
| 測試集評估結果 | [output/evaluation.json](output/evaluation.json) | 準確率、混淆矩陣、兩類precision／recall／F1、Macro-F1、三折驗證與基準 |
| 分析文本逐筆預測 | [output/application_predictions.csv](output/application_predictions.csv) | 100則應用文本的預測標籤與機率 |
| 分析文本分組摘要 | [output/application_group_summary.csv](output/application_group_summary.csv) | 各原細類與二元參考標籤的一致程度 |
| 分析閱讀頁（選做） | [output/results.html](output/results.html)、[output/application_review.html](output/application_review.html) | 評估、權重與可點擊展開的E／A案例；下載後用瀏覽器開啟 |
| 訓練好的模型 | [models/cold_logistic.joblib](models/cold_logistic.joblib) | 包含詞彙表、IDF及邏輯迴歸的完整Pipeline |
| 參數與套件版本 | [output/run_metadata.json](output/run_metadata.json)、[requirements.txt](requirements.txt) | 種子、資料量、版本與原始檔SHA-256 |
| 最後核查 | [output/final_audit.json](output/final_audit.json) | 樣本量、資料分離、指標、模型預測及必要檔案檢查 |

HTML頁面可展開案例，但**尚未部署為GitHub Pages網站，也不是線上即時分類器**。GitHub通常只顯示HTML原始碼；請選 **Code → Download ZIP**，解壓後用瀏覽器開啟 `output/results.html`。GitHub Pages屬於指引的選做項目。

## 文中A01、E01、B13等例子在哪裏？

案例代碼是本研究的閱讀編號，不是原資料集ID。可在下列檔案搜尋 `A01` 等代碼：

| 代碼 | 原文、來源及模型資料 | 學生判讀／閱讀頁 |
|---|---|---|
| A01–A06：新文本應用 | [review/application_cases.json](review/application_cases.json) | [review/user_application_review.json](review/user_application_review.json)、[output/application_review.html](output/application_review.html) |
| E01–E08：保留測試集分歧 | [review/error_cases.json](review/error_cases.json) | [review/user_error_review.json](review/user_error_review.json)、[output/results.html](output/results.html) |
| B01–B28：訓練資料抽查 | [review/training_audit_28.json](review/training_audit_28.json) | [review/user_training_audit_28.json](review/user_training_audit_28.json) |
| R01–R12：前期規則試標 | [review/reference_labels.json](review/reference_labels.json) | [review/user_review_round1.json](review/user_review_round1.json) |

例如 **A01** 在 `review/application_cases.json` 的 `case_id` 欄，對應 `source_row=604`，來源為 [data/raw/test.csv](data/raw/test.csv)。也可在 `output/application_predictions.csv` 以 `source_file=test.csv`、`source_row=604` 找到預測。E及B案例來自 `data/raw/train.csv`。`source_row`是包含標題列的CSV記錄序號；若文本內有換行，不等同純文字編輯器的物理行數。JSON保留原文，報告引文轉為繁體。

## 標籤、文本單位與標註規則

一則完整留言為一筆資料，不將同一留言拆散至不同集合。**1＝冒犯**：對個人或群體實施或認同侮辱、貶抑等冒犯。**0＝非冒犯**：未實施上述冒犯，包括一般批評、描述或反對冒犯。這是操作性概括，建模沿用COLD標註。

判讀須分清實施、認同、轉述與反對，不能只看粗口或群體名稱。「非冒犯」不等於「反偏見」；後者須有可辨認的反對、糾正或辯護對象。U只表示人工審核不確定，不是第三種模型標籤。詳細規則見 [review/annotation_guidelines.md](review/annotation_guidelines.md)。

1,000筆標籤沿用原資料集，並非全部由學生或AI從零標註。學生完成12則規則試標、28則訓練留言抽查、8則測試案例及6則應用細讀；分歧另存，不事後改標提高測試分數。

## 資料來源與方法

來源：[COLD官方倉庫](https://github.com/thu-coai/COLDataset)、[原論文](https://aclanthology.org/2022.emnlp-main.796/)。本次使用學生提供的官方CSV副本；原始檔見 [data/raw/](data/raw/)，授權見 [data/COLD-LICENSE.txt](data/COLD-LICENSE.txt)。兩份相同test下載只保留一份。

去空白比對後排除27組跨來源同文異標、精確重複、官方test同文及12則規則開發留言，從官方train每類抽500筆。按標籤分層80/20切分，種子42。官方dev僅作重複及標籤衝突稽核。

jieba分詞、移除純標點，不作繁簡轉換、不套通用停用詞清單。單詞TF-IDF（min_df=2）及 LogisticRegression（C=1、max_iter=2000、random_state=42），分類門檻0.5。三折交叉驗證只用800筆訓練資料，各折重新學習詞彙表及IDF。固定單一模型，未依測試結果調參。保留測試準確率82%、Macro-F1約0.819。

官方test按細類0其他非冒犯、1攻擊個人、2攻擊群體、3反偏見各抽25則作應用。二元一致率61%；反偏見組44%不是四類分類準確率。同來源及平衡抽樣限制泛化。[近重複檢查](output/near_duplicate_audit.json)以字元3–5元組餘弦相似度0.65篩查，未找到跨集合候選，但不能排除所有語意改寫。

## 如何重跑程式

使用Python 3.12，在專案根目錄執行；建議先建立獨立虛擬環境：

```bash
python -m pip install -r requirements.txt
python scripts/run_pipeline.py
python scripts/audit_data.py
python scripts/build_results.py
python scripts/build_application_review.py
python scripts/render_report.py
python scripts/verify_project.py
```

`run_pipeline.py`重建切分、模型及預測；`audit_data.py`產生近重複檢查及B案例工作紙；`build_results.py`重新載入模型核對預測，產生E案例頁及應用分組摘要；`build_application_review.py`產生A案例頁。人工答案獨立保存在 `user_*.json`，上述程式不改寫它們。

`render_report.py`由同一份Markdown產生PDF及 [報告閱讀頁](output/report.html)，使用Windows微軟正黑體 `C:/Windows/Fonts/msjh.ttc`；其他系統須調整字體路徑。Markdown標記只用於格式，不印在PDF內。字數口徑及數值見 [output/report_check.json](output/report_check.json)。

## AI使用及提交

學生負責方向與計劃討論、人工判讀、提出概念疑問並修改報告。Codex協助程式、資料稽核、訓練、解釋及依學生筆記整合初稿；完整聲明見 [report.md](report.md)。倉庫不列學生姓名或學號。

Moodle須另交公開倉庫網址與同一份 `report.pdf`；GitHub上傳不等於Moodle提交。暫存圖片、快取及舊版閱讀頁不納入倉庫。

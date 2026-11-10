"""Verify assignment deliverables without changing reference labels or fitting a model."""
from pathlib import Path
import hashlib
import json
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]


def main():
    required = ['README.md', 'report.md', 'report.pdf', 'requirements.txt',
                'data/labeled_data.csv', 'data/train_split.csv', 'data/test_split.csv',
                'data/application_data.csv', 'scripts/run_pipeline.py', 'scripts/text_features.py',
                'models/cold_logistic.joblib', 'output/evaluation.json',
                'output/test_predictions.csv', 'output/application_predictions.csv',
                'output/results.html', 'output/application_review.html']
    assert all((ROOT / p).is_file() for p in required)
    frames = {name: pd.read_csv(ROOT / f'data/{name}.csv') for name in
              ['labeled_data', 'train_split', 'test_split', 'application_data']}
    expected = {'labeled_data': {0: 500, 1: 500}, 'train_split': {0: 400, 1: 400},
                'test_split': {0: 100, 1: 100}, 'application_data': {0: 50, 1: 50}}
    keys = {}
    for name, df in frames.items():
        assert df.label.value_counts().to_dict() == expected[name]
        assert df.text.notna().all()
        keys[name] = set(df.text.map(lambda s: re.sub(r'\s+', '', s)))
        assert len(keys[name]) == len(df)
    assert keys['train_split'] | keys['test_split'] == keys['labeled_data']
    assert not keys['train_split'] & keys['test_split']
    assert not keys['labeled_data'] & keys['application_data']
    metadata = json.loads((ROOT / 'output/run_metadata.json').read_text(encoding='utf-8'))
    for name, digest in metadata['raw_sha256'].items():
        assert hashlib.sha256((ROOT / f'data/raw/{name}.csv').read_bytes()).hexdigest() == digest
    model = joblib.load(ROOT / 'models/cold_logistic.joblib')
    assert model.named_steps['classifier'].__class__.__name__ == 'LogisticRegression'
    assert metadata['seed'] == 42
    predictions = {}
    for name, split in [('test', 'test_split'), ('application', 'application_data')]:
        df = pd.read_csv(ROOT / f'output/{name}_predictions.csv')
        assert df.text.tolist() == frames[split].text.tolist()
        assert df.label.tolist() == frames[split].label.tolist()
        assert np.array_equal(model.predict(df.text), df.prediction)
        assert np.allclose(model.predict_proba(df.text)[:, 1], df.p_offensive)
        assert np.array_equal(df.label == df.prediction, df.matches_reference)
        predictions[name] = df
    metrics = json.loads((ROOT / 'output/evaluation.json').read_text(encoding='utf-8'))
    test = predictions['test']
    assert accuracy_score(test.label, test.prediction) == metrics['accuracy']
    assert confusion_matrix(test.label, test.prediction, labels=[0, 1]).tolist() == metrics['confusion_matrix_rows_true_columns_predicted_0_1']
    recalculated = classification_report(test.label, test.prediction, output_dict=True)
    for label in ['0', '1', 'macro avg', 'weighted avg']:
        for stat in ['precision', 'recall', 'f1-score', 'support']:
            assert np.isclose(recalculated[label][stat], metrics['classification_report'][label][stat])
    assert len(metrics['training_only_cv_macro_f1']) == 3
    for filename, predname in [('error_cases.json', 'test'), ('application_cases.json', 'application')]:
        for case in json.loads((ROOT / 'review' / filename).read_text(encoding='utf-8')):
            match = predictions[predname].loc[lambda d: d.source_row == case['source_row']]
            assert len(match) == 1 and match.iloc[0].text == case['text']
            assert int(match.iloc[0].prediction) == case['prediction']
    md = (ROOT / 'report.md').read_text(encoding='utf-8-sig')
    count = len(re.findall(r'[\u4e00-\u9fff]', md.split('## 參考資料')[0]))
    assert 2000 <= count <= 2500
    readme = (ROOT / 'README.md').read_text(encoding='utf-8-sig')
    links = re.findall(r'\]\(([^)]+)\)', readme)
    for target in links:
        if not target.startswith(('https://', 'http://', '#')) and target != 'output/final_audit.json':
            assert (ROOT / target).exists(), target
    summary = {'status': 'passed', 'required_files_present': required,
               'class_counts': {k: {str(a): b for a, b in v.items()} for k, v in expected.items()},
               'exact_duplicates_and_cross_set_overlap': 0, 'saved_model_predictions_verified': 300,
               'accuracy': metrics['accuracy'], 'confusion_matrix': metrics['confusion_matrix_rows_true_columns_predicted_0_1'],
               'application_reference_agreement': float(predictions['application'].matches_reference.mean()),
               'report_han_characters_excluding_references_and_AI': count,
               'raw_hashes_match': True, 'case_source_mappings_match': True,
               'readme_local_links_exist': True,
               'limitations': ['Same-source application; not cross-platform validation.',
                              'Existing COLD labels with sampled student review, not all newly annotated.',
                              'HTML exploration pages provided; optional GitHub Pages not deployed.',
                              'Moodle upload remains separate.']}
    (ROOT / 'output/final_audit.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()

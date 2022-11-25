import json
import click
from pdf_struct import loader
from pdf_struct.core import transition_labels
from pdf_struct.core.evaluation import evaluate
from pdf_struct import feature_extractor

def _evaluate(k_folds: int, prediction, metrics, file_type: str,feature : str, raw_dir, anno_dir):
    print(f'Loading annotations from {anno_dir}')
    annos = transition_labels.load_annos(anno_dir)
    print('Loading raw files')
    documents = loader.modules[file_type].load_from_directory(raw_dir, annos)
    feature_extractor_cls = feature_extractor.feature_extractors[feature]
    if len(documents) == 0:
        click.echo(
            f'No matching documents found in {raw_dir}. Please check directory '
            'path and file extention.', err=True)
        exit(1)
    if prediction is not None:
        metrics_, prediction = evaluate(
            documents,feature_extractor_cls, k_folds, True)
        with open(prediction, 'w') as fout:
            for p in prediction:
                fout.write(json.dumps(p))
                fout.write('\n')
    else:
        metrics_ = evaluate(documents,feature_extractor_cls, k_folds, False)

    if metrics is None:
        print(json.dumps(metrics_, indent=2))
    else:
        with open(metrics, 'w') as fout:
            json.dump(metrics, fout, indent=2)

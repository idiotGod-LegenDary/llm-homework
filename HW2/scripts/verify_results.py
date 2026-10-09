"""Recompute scores from saved predictions; verify provenance and split invariants."""
import csv,hashlib,json,math
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results'
def read_csv(name):
    with (R/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def read_json(name):return json.loads((R/name).read_text(encoding='utf8'))
def check(condition,message):
    if not condition:raise AssertionError(message)
labels=read_json('classifier_config.json')['labels']
manifest=read_csv('document_manifest.csv')
check(len({r['uid'] for r in manifest})==len(manifest),'duplicate IDs')
check(len({r['body_sha256'] for r in manifest})==len(manifest),'duplicate body hashes')
check(len({r['prefix_sha256'] for r in manifest})==len(manifest),'duplicate prefixes')
check(all(r['representation']=='False' for r in manifest if r['split']=='test'),'test documents in embedding training')
check(Counter(r['label'] for r in manifest if r['split']=='train')==Counter({l:48 for l in labels}),'train counts')
check(Counter(r['label'] for r in manifest if r['split']=='test')==Counter({l:12 for l in labels}),'test counts')
groups=defaultdict(list)
for r in read_csv('predictions.csv'):groups[(int(r['classes']),r['initialization'],int(r['seed']))].append(r)
metrics=read_csv('classification_metrics.csv');reports=read_json('classification_reports.json')
check(len(groups)==18 and len(metrics)==18,'wrong condition count')
for row in metrics:
    key=(int(row['classes']),row['initialization'],int(row['seed']));preds=groups[key];active=labels[:key[0]]
    check(len(preds)==key[0]*12,'test count')
    check(len({p['uid'] for p in preds})==len(preds),'duplicate predictions')
    matrix=[[0 for _ in active] for _ in active]
    for p in preds:matrix[active.index(p['actual'])][active.index(p['predicted'])]+=1
    correct=sum(matrix[i][i] for i in range(len(active)))
    f1=[]
    for i in range(len(active)):
        tp=matrix[i][i];fp=sum(matrix[j][i] for j in range(len(active)) if j!=i);fn=sum(matrix[i][j] for j in range(len(active)) if j!=i)
        f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    check(correct==int(row['correct']),'correct count')
    check(math.isclose(correct/len(preds),float(row['accuracy']),abs_tol=1e-12),'accuracy')
    check(math.isclose(sum(f1)/len(f1),float(row['macro_f1']),abs_tol=1e-12),'macro F1')
    check(matrix==reports[f'{key[0]}_{key[1]}_{key[2]}']['confusion_matrix'],'confusion matrix')
    for small in [2,4]:
        if small>=key[0]:continue
        smaller=groups[(small,key[1],key[2])]
        check({p['uid'] for p in smaller}<={p['uid'] for p in preds},'non-nested test IDs')
primary={p['uid']:p['predicted'] for p in groups[(8,'pretrained',42)]}
repeat=read_json('classification_repeat.json')
check(repeat['test_documents']==96 and repeat['identical_predictions'] and repeat['matching_predictions']==96,'training script reported a repetition mismatch')
for entry in read_json('download_manifest.json')['files']:
    path=ROOT/'data'/'raw'/(entry['label']+'.jsonl')
    if path.exists():check(hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256'],'raw prefix changed')
for filename,value in read_json('data_statistics.json')['data_sha256'].items():
    path=ROOT/'data'/'processed'/filename
    if path.exists():check(hashlib.sha256(path.read_bytes()).hexdigest()==value,'processed input changed')

convergence=read_csv('convergence_metrics.csv');diagnostic_groups=defaultdict(list)
for p in read_csv('convergence_predictions.csv'):diagnostic_groups[(p['initialization'],int(p['epoch']),p['split'])].append(p)
convergence_reports=read_json('convergence_reports.json')
for m in convergence:
    init=m['initialization'];epoch=int(m['epoch'])
    for split in ['train','test']:
        rows=diagnostic_groups[(init,epoch,split)]
        correct=sum(p['actual']==p['predicted'] for p in rows)
        check(correct==int(m[split+'_correct']) and len(rows)==int(m[split+'_count']),'convergence counts')
        check(math.isclose(correct/len(rows),float(m[split+'_accuracy']),abs_tol=1e-12),'convergence accuracy')
        if split=='test':
            matrix=[[0]*8 for _ in range(8)]
            for p in rows:matrix[labels.index(p['actual'])][labels.index(p['predicted'])]+=1
            f1=[]
            for i in range(8):
                tp=matrix[i][i];fp=sum(matrix[j][i] for j in range(8) if j!=i);fn=sum(matrix[i][j] for j in range(8) if j!=i)
                f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
            check(math.isclose(sum(f1)/8,float(m['test_macro_f1']),abs_tol=1e-12),'convergence macro F1')
            check(matrix==convergence_reports[f'{init}_{epoch}']['confusion_matrix'],'convergence matrix')
            check({p['uid'] for p in rows}==set(primary),'convergence changed test set')
            if epoch==25:
                original={p['uid']:p['predicted'] for p in groups[(8,init,42)]}
                check(all(original[p['uid']]==p['predicted'] for p in rows),'saved model differs from primary run')

result={'status':'passed','metrics_recomputed':len(metrics),'diagnostic_conditions_recomputed':len(convergence),'diagnostic_test_rows':sum(len(v) for k,v in diagnostic_groups.items() if k[2]=='test'),'diagnostic_train_rows':sum(len(v) for k,v in diagnostic_groups.items() if k[2]=='train'),'primary_seed':42,'prediction_rows':sum(len(v) for v in groups.values()),'repeat_rows':repeat['test_documents'],'repeat_check':'per-prediction equality asserted inside the training process; metadata checked here','independent_test_documents':96,'split_checks':'unique IDs, body hashes and prefix hashes; 48/12 per category; nested test subsets; all test excluded from representation','verification':'independent arithmetic reconstruction of accuracy, macro F1 and confusion matrices from predictions'}
(R/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(result,ensure_ascii=False,indent=2))

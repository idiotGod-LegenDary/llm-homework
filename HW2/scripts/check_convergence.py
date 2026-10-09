"""A targeted follow-up: check whether the weak scratch baseline is undertrained."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
import csv,gc,json,time
from pathlib import Path
import fasttext
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT);R=ROOT/'results'
labels=json.loads((R/'classifier_config.json').read_text(encoding='utf8'))['labels']
settings=json.loads((R/'classifier_config.json').read_text(encoding='utf8'))['common']
def examples(split):
    rows=[]
    for line in (ROOT/'data/processed'/f'8_{split}.txt').read_text(encoding='utf8').splitlines():
        label,text=line.split(' ',1);rows.append((label.replace('__label__',''),text))
    return rows
train=examples('train');test=examples('test')
with (R/'predictions.csv').open(encoding='utf-8-sig') as f:
    ids=[r['uid'] for r in csv.DictReader(f) if r['classes']=='8' and r['seed']=='42' and r['initialization']=='pretrained']
assert len(ids)==len(test)==96
metrics=[];predictions=[];reports={};total=time.perf_counter()
for initialization,epoch in [('pretrained',25),('scratch',25),('scratch',100),('scratch',250)]:
    seconds=0.
    if epoch==25:model=fasttext.load_model(f'models/classification_8_{initialization}.bin')
    else:
        config=dict(settings,epoch=epoch,seed=42)
        start=time.perf_counter();model=fasttext.train_supervised(input='data/processed/8_train.txt',**config);seconds=time.perf_counter()-start
    start=time.perf_counter();train_predictions=[model.predict(text)[0][0].replace('__label__','') for _,text in train];test_predictions=[model.predict(text)[0][0].replace('__label__','') for _,text in test];infer_seconds=time.perf_counter()-start
    actual=[label for label,_ in test]
    row={'initialization':initialization,'epoch':epoch,'seed':42,'new_training':epoch!=25,'train_correct':sum(a==b for (a,_),b in zip(train,train_predictions)),'train_count':len(train),'train_accuracy':float(accuracy_score([a for a,_ in train],train_predictions)),'test_correct':sum(a==b for a,b in zip(actual,test_predictions)),'test_count':len(test),'test_accuracy':float(accuracy_score(actual,test_predictions)),'test_macro_f1':float(f1_score(actual,test_predictions,labels=labels,average='macro',zero_division=0)),'train_seconds':seconds,'evaluation_seconds':infer_seconds}
    metrics.append(row);print('CONVERGENCE',json.dumps(row,ensure_ascii=False),flush=True)
    for split,items,preds,uids in [('train',train,train_predictions,[str(i) for i in range(len(train))]),('test',test,test_predictions,ids)]:
        for (a,_),b,uid in zip(items,preds,uids):predictions.append({'initialization':initialization,'epoch':epoch,'seed':42,'split':split,'uid':uid,'actual':a,'predicted':b,'correct':a==b})
    reports[f'{initialization}_{epoch}']={'labels':labels,'confusion_matrix':confusion_matrix(actual,test_predictions,labels=labels).tolist()}
    del model;gc.collect()
for name,rows in [('convergence_metrics.csv',metrics),('convergence_predictions.csv',predictions)]:
    with (R/name).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
(R/'convergence_reports.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2),encoding='utf8')
summary={'reason':'Scratch baseline was weak; check training budget without retuning the primary comparison','additional_trainings':2,'epochs':[100,250],'seed':42,'fixed_test_documents':96,'prediction_rows':len(predictions),'test_prediction_rows':4*len(test),'train_prediction_rows':4*len(train),'seconds':time.perf_counter()-total,'new_training_seconds':sum(r['train_seconds'] for r in metrics)}
(R/'convergence_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
print('CONVERGENCE_COMPLETE',flush=True)

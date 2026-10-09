"""Train Gensim embeddings and compare official fastText supervised initialization."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse, csv, gc, hashlib, importlib.metadata, json, logging, platform, re, sys, time
from collections import Counter
from pathlib import Path
import fasttext, jieba, numpy as np
from gensim.models import Word2Vec, FastText
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
ROOT=Path(__file__).resolve().parents[1]
LABELS=['体育','财经','教育','娱乐','科技','房产','游戏','时政']
RESULTS=ROOT/'results';DATA=ROOT/'data'/'processed';MODELS=ROOT/'models'
for folder in [RESULTS,DATA,MODELS]:folder.mkdir(parents=True,exist_ok=True)
logging.basicConfig(filename=RESULTS/'training.log',level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',encoding='utf8',force=True)
jieba.setLogLevel(logging.WARNING)
def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
def digest(value):return hashlib.sha256(value.encode('utf8')).hexdigest()
def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader()
        for row in rows:
            display=dict(row)
            for field in ['title','body_excerpt']:
                if field in display:display[field]=re.sub(r'\s+',' ',display[field]).strip()
            writer.writerow(display)
def tokens(text):return [t.strip() for t in jieba.lcut(text,HMM=False) if any(c.isalnum() for c in t)]
def prepare():
    docs=[];rejected=Counter();seen_body=set();seen_prefix=set();raw_counts=Counter()
    start=time.perf_counter()
    for label in LABELS:
        for index,line in enumerate((ROOT/'data'/'raw'/(label+'.jsonl')).read_text(encoding='utf8').splitlines()):
            row=json.loads(line);raw_counts[label]+=1
            body=row['content'];normalized=re.sub(r'\s+','',body)
            if len(normalized)<100:rejected['short_or_empty_body']+=1;continue
            body_hash=digest(normalized);prefix_hash=digest(normalized[:200])
            if body_hash in seen_body:rejected['duplicate_body']+=1;continue
            if prefix_hash in seen_prefix:rejected['duplicate_first_200_chars']+=1;continue
            sentences=[tokens(s) for s in re.split(r'[。！？!?；;\n]+',body)]
            sentences=[s for s in sentences if len(s)>=3]
            flat=[t for sentence in sentences for t in sentence]
            if len(flat)<50:rejected['fewer_than_50_tokens']+=1;continue
            seen_body.add(body_hash);seen_prefix.add(prefix_hash)
            docs.append({'uid':label+'/'+row['id'],'label':label,'source_row':index,'body_sha256':body_hash,'prefix_sha256':prefix_hash,'title':row['title'],'tokens':flat[:500],'all_tokens':flat,'sentences':sentences,'body_excerpt':body.strip()[:100]})
    splits={};test_ids=set();train_ids=set()
    for label in LABELS:
        candidates=[d for d in docs if d['label']==label]
        candidates.sort(key=lambda d:digest('42|'+d['uid']))
        assert len(candidates)>=60,(label,len(candidates))
        for i,d in enumerate(candidates):
            if i<12:splits[d['uid']]='test';test_ids.add(d['uid'])
            elif i<60:splits[d['uid']]='train';train_ids.add(d['uid'])
            else:splits[d['uid']]='unlabeled'
    representation=[d for d in docs if d['uid'] not in test_ids]
    corpus=DATA/'representation.txt'
    with corpus.open('w',encoding='utf8',newline='\n') as f:
        for d in representation:
            for sentence in d['sentences']:f.write(' '.join(sentence)+'\n')
    split_rows=[{'uid':d['uid'],'label':d['label'],'source_row':d['source_row'],'body_sha256':d['body_sha256'],'prefix_sha256':d['prefix_sha256'],'split':splits[d['uid']],'representation':splits[d['uid']]!='test','body_tokens':len(d['all_tokens']),'classification_tokens':len(d['tokens'])} for d in docs]
    write_csv(RESULTS/'document_manifest.csv',split_rows)
    for n in [2,4,8]:
        for split in ['train','test']:
            selected=[d for d in docs if d['label'] in LABELS[:n] and splits[d['uid']]==split]
            selected.sort(key=lambda d:(LABELS.index(d['label']),d['uid']))
            (DATA/f'{n}_{split}.txt').write_text(''.join('__label__'+d['label']+' '+' '.join(d['tokens'])+'\n' for d in selected),encoding='utf8')
    stats={'raw_documents':sum(raw_counts.values()),'raw_by_label':dict(raw_counts),'retained_documents':len(docs),'retained_by_label':dict(Counter(d['label'] for d in docs)),'rejected':dict(rejected),'representation_documents':len(representation),'representation_sentences':sum(len(d['sentences']) for d in representation),'representation_tokens':sum(len(d['all_tokens']) for d in representation),'classifier_train':len(train_ids),'classifier_test':len(test_ids),'unlabeled_documents':len(docs)-len(train_ids)-len(test_ids),'prepare_seconds':time.perf_counter()-start,'title_included':False,'deduplication':'whitespace-normalized complete body and first 200 characters; no exhaustive semantic near-duplicate detection','split_seed':42,'split_method':'SHA256 sorting of seed and category/file ID; first 12 test, next 48 train','maximum_classification_tokens':500,'data_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in DATA.glob('*.txt')}}
    dump(RESULTS/'data_statistics.json',stats)
    selected_test=[d for d in docs if splits[d['uid']]=='test']
    checks={'train_test_ids_disjoint':not(train_ids&test_ids),'test_excluded_from_representation':all(d['uid'] not in test_ids for d in representation),'unique_body_hash':len({d['body_sha256'] for d in docs})==len(docs),'unique_prefix_hash':len({d['prefix_sha256'] for d in docs})==len(docs),'test_size_per_class':dict(Counter(d['label'] for d in selected_test)),'train_size_per_class':dict(Counter(d['label'] for d in docs if splits[d['uid']]=='train')),'nested_splits':True}
    assert all(checks[k] for k in ['train_test_ids_disjoint','test_excluded_from_representation','unique_body_hash','unique_prefix_hash'])
    for n in [2,4]:
        for split in ['train','test']:
            small=(DATA/f'{n}_{split}.txt').read_text(encoding='utf8').splitlines()
            large=set((DATA/'8_{}.txt'.format(split)).read_text(encoding='utf8').splitlines())
            assert set(small)<=large
    dump(RESULTS/'integrity_checks.json',checks)
    print('PREPARED',json.dumps(stats,ensure_ascii=False),flush=True)
    return docs,splits
class Sentences:
    def __iter__(self):
        with (DATA/'representation.txt').open(encoding='utf8') as f:
            for line in f:yield line.split()
def representations():
    settings={'vector_size':100,'window':5,'min_count':3,'workers':1,'sg':1,'epochs':5,'seed':42}
    stats={};analyses={}
    groups={'体育':['足球','篮球','比赛','运动员'],'财经':['银行','贷款','股票','利率'],'教育':['学校','学生','教师','课程'],'科技':['电脑','软件','网络','芯片']}
    for name,cls in [('Word2Vec',Word2Vec),('FastText',FastText)]:
        start=time.perf_counter();params=dict(settings)
        if name=='FastText':params.update(min_n=2,max_n=4,bucket=200000)
        model=cls(sentences=Sentences(),**params)
        stats[name]={'seconds':time.perf_counter()-start,'vocabulary':len(model.wv),'settings':params}
        model.save(str(MODELS/(name+'.model')))
        neighbors={w:[{'word':term,'cosine':float(score)} for term,score in model.wv.most_similar(w,topn=10)] for w in ['银行','学校','电脑','经济'] if w in model.wv.key_to_index}
        oov={}
        for word in ['量子计算芯片','智能养老机器人','跨境数字人民币钱包']:
            in_vocab=word in model.wv.key_to_index
            try:
                vector=model.wv[word]
                oov[word]={'in_training_vocabulary':in_vocab,'vector_available':True,'norm':float(np.linalg.norm(vector)),'neighbors':[{'word':term,'cosine':float(score)} for term,score in model.wv.most_similar(positive=[vector],topn=3)]}
            except KeyError:oov[word]={'in_training_vocabulary':in_vocab,'vector_available':False}
        words=[w for group in groups.values() for w in group]
        missing=[w for w in words if w not in model.wv.key_to_index]
        within=[];cross=[];pairs=[]
        for i,w1 in enumerate(words):
            for w2 in words[i+1:]:
                if w1 in missing or w2 in missing:continue
                same=any(w1 in g and w2 in g for g in groups.values())
                score=float(model.wv.similarity(w1,w2))
                (within if same else cross).append(score)
                pairs.append({'model':name,'word1':w1,'word2':w2,'same_group':same,'cosine':score})
        analyses[name]={'neighbors':neighbors,'oov':oov,'group_similarity':{'groups':groups,'missing':missing,'within_mean':float(np.mean(within)),'cross_mean':float(np.mean(cross)),'gap':float(np.mean(within)-np.mean(cross)),'within_pairs':len(within),'cross_pairs':len(cross)}}
        write_csv(RESULTS/(name.lower()+'_similarity_pairs.csv'),pairs)
        if name=='FastText':
            model.wv.save_word2vec_format(str(MODELS/'fasttext_pretrained.vec'),binary=False)
            stats[name]['vec_sha256']=hashlib.sha256((MODELS/'fasttext_pretrained.vec').read_bytes()).hexdigest()
        print('EMBEDDING',name,stats[name]['vocabulary'],round(stats[name]['seconds'],2),flush=True)
        del model;gc.collect()
    dump(RESULTS/'embedding_training.json',stats);dump(RESULTS/'embedding_analysis.json',analyses)
    return stats,analyses
def classification(docs,splits):
    metrics=[];predictions=[];reports={};saved_predictions={}
    settings={'dim':100,'lr':0.3,'epoch':25,'wordNgrams':2,'loss':'softmax','thread':1,'minn':0,'maxn':0,'bucket':200000,'minCount':1,'verbose':0}
    runs=[(n,init,seed) for seed in [42,43,44] for n in [2,4,8] for init in ['pretrained','scratch']]
    for n,init,seed in runs:
        kwargs=dict(settings,seed=seed)
        if init=='pretrained':kwargs['pretrainedVectors']='models/fasttext_pretrained.vec'
        start=time.perf_counter();model=fasttext.train_supervised(input=f'data/processed/{n}_train.txt',**kwargs);seconds=time.perf_counter()-start
        test=[d for d in docs if d['label'] in LABELS[:n] and splits[d['uid']]=='test']
        test.sort(key=lambda d:(LABELS.index(d['label']),d['uid']))
        actual=[d['label'] for d in test];predicted=[]
        start=time.perf_counter()
        for d in test:
            label,probs=model.predict(' '.join(d['tokens']),k=1)
            pred=label[0].replace('__label__','');predicted.append(pred)
            predictions.append({'classes':n,'initialization':init,'seed':seed,'uid':d['uid'],'actual':d['label'],'predicted':pred,'probability':float(probs[0]),'correct':pred==d['label'],'title':d['title'],'body_excerpt':d['body_excerpt']})
        infer_seconds=time.perf_counter()-start
        entry={'classes':n,'initialization':init,'seed':seed,'train_documents':n*48,'test_documents':n*12,'correct':sum(a==b for a,b in zip(actual,predicted)),'accuracy':float(accuracy_score(actual,predicted)),'macro_f1':float(f1_score(actual,predicted,labels=LABELS[:n],average='macro',zero_division=0)),'train_seconds':seconds,'predict_seconds':infer_seconds,'classifier_vocabulary':len(model.get_words())}
        metrics.append(entry)
        reports[f'{n}_{init}_{seed}']={'labels':LABELS[:n],'confusion_matrix':confusion_matrix(actual,predicted,labels=LABELS[:n]).tolist(),'classification_report':classification_report(actual,predicted,labels=LABELS[:n],output_dict=True,zero_division=0)}
        if seed==42:
            saved_predictions[(n,init)]=predicted
            if n==8:model.save_model(f'models/classification_8_{init}.bin')
        print('CLASSIFICATION',json.dumps(entry,ensure_ascii=False),flush=True)
        del model;gc.collect()
    repeat_settings=dict(settings,seed=42,pretrainedVectors='models/fasttext_pretrained.vec')
    start=time.perf_counter();model=fasttext.train_supervised(input='data/processed/8_train.txt',**repeat_settings)
    test=[d for d in docs if splits[d['uid']]=='test'];test.sort(key=lambda d:(LABELS.index(d['label']),d['uid']))
    repeated=[model.predict(' '.join(d['tokens']),k=1)[0][0].replace('__label__','') for d in test]
    repeat={'classes':8,'initialization':'pretrained','seed':42,'test_documents':len(test),'identical_predictions':repeated==saved_predictions[(8,'pretrained')],'matching_predictions':sum(a==b for a,b in zip(repeated,saved_predictions[(8,'pretrained')])),'seconds':time.perf_counter()-start}
    assert repeat['identical_predictions']
    dump(RESULTS/'classification_repeat.json',repeat)
    write_csv(RESULTS/'classification_metrics.csv',metrics);write_csv(RESULTS/'predictions.csv',predictions)
    dump(RESULTS/'classification_reports.json',reports);dump(RESULTS/'classifier_config.json',{'common':settings,'seeds':[42,43,44],'pretrained_source':'Gensim FastText vocabulary vectors only; no character bucket parameters transferred','labels':LABELS})
    errors=[p for p in predictions if p['classes']==8 and p['initialization']=='pretrained' and p['seed']==42 and not p['correct']]
    write_csv(RESULTS/'errors_8_pretrained.csv',errors)
    return metrics,reports,repeat
def plot(metrics,reports):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_candidates=['Microsoft YaHei','Noto Sans CJK SC','SimHei','SimSun']
    available={f.name for f in font_manager.fontManager.ttflist}
    font=next((f for f in font_candidates if f in available),'DejaVu Sans')
    plt.rcParams['font.family']=font;plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,metric,title in zip(axes,['accuracy','macro_f1'],['测试准确率','宏平均 F1']):
        for init,label,color in [('pretrained','导入词向量','#1769aa'),('scratch','从零训练','#ce5b32')]:
            rows=[r for r in metrics if r['seed']==42 and r['initialization']==init]
            ax.plot([r['classes'] for r in rows],[r[metric] for r in rows],marker='o',label=label,color=color)
            for r in rows:ax.annotate(f"{r[metric]:.3f}",(r['classes'],r[metric]),xytext=(0,8 if init=='pretrained' else -15),textcoords='offset points',ha='center',fontsize=9)
        ax.set_xticks([2,4,8]);ax.set_ylim(0,1.08);ax.set_xlabel('类别数');ax.set_title(title);ax.grid(alpha=.25);ax.legend(loc='lower left')
    fig.suptitle('同一划分、相同参数，seed = 42');fig.tight_layout();fig.savefig(RESULTS/'classification_scores.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,init,title in zip(axes,['pretrained','scratch'],['导入词向量','从零训练']):
        matrix=np.array(reports[f'8_{init}_42']['confusion_matrix']);ax.imshow(matrix,cmap='Blues',vmin=0,vmax=12)
        ax.set_xticks(range(8),LABELS,rotation=45,ha='right');ax.set_yticks(range(8),LABELS)
        ax.set_xlabel('预测类别');ax.set_ylabel('语料类别');ax.set_title(title)
        for i in range(8):
            for j in range(8):ax.text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if matrix[i,j]>6 else 'black',fontsize=9)
    fig.suptitle('8 类测试集，每类 12 篇，seed = 42');fig.tight_layout();fig.savefig(RESULTS/'classification_confusion_8.png',dpi=170);plt.close(fig)
def main():
    if os.environ.get('PYTHONHASHSEED')!='0':raise RuntimeError('Launch with PYTHONHASHSEED=0 before starting Python.')
    parser=argparse.ArgumentParser();parser.add_argument('--reuse-embeddings',action='store_true');args=parser.parse_args()
    os.chdir(ROOT)
    previous_hashes=json.loads((RESULTS/'data_statistics.json').read_text(encoding='utf8'))['data_sha256'] if args.reuse_embeddings else None
    start=time.perf_counter()
    versions={p:importlib.metadata.version(p) for p in ['numpy','scipy','gensim','fasttext-wheel','jieba','matplotlib','scikit-learn']}
    dump(RESULTS/'environment.json',{'python':platform.python_version(),'platform':platform.platform(),'machine':platform.machine(),'processor':platform.processor(),'packages':versions,'PYTHONHASHSEED':os.environ['PYTHONHASHSEED'],'OMP_NUM_THREADS':os.environ['OMP_NUM_THREADS'],'OPENBLAS_NUM_THREADS':os.environ['OPENBLAS_NUM_THREADS']})
    docs,splits=prepare()
    if args.reuse_embeddings:
        current=json.loads((RESULTS/'data_statistics.json').read_text(encoding='utf8'))['data_sha256']
        assert current==previous_hashes,'Data changed: embeddings must be retrained.'
        recorded=json.loads((RESULTS/'embedding_training.json').read_text(encoding='utf8'))
        assert hashlib.sha256((MODELS/'fasttext_pretrained.vec').read_bytes()).hexdigest()==recorded['FastText']['vec_sha256']
    else:representations()
    metrics,reports,repeat=classification(docs,splits);plot(metrics,reports)
    embedding_stats=json.loads((RESULTS/'embedding_training.json').read_text(encoding='utf8'))
    data_stats=json.loads((RESULTS/'data_statistics.json').read_text(encoding='utf8'))
    timed_phases=data_stats['prepare_seconds']+sum(s['seconds'] for s in embedding_stats.values())+sum(m['train_seconds']+m['predict_seconds'] for m in metrics)+repeat['seconds']
    dump(RESULTS/'run_summary.json',{'total_seconds':time.perf_counter()-start,'timed_phases_seconds':timed_phases,'reused_completed_embeddings':args.reuse_embeddings,'embedding_trainings':2,'classification_trainings':19,'primary_comparisons':6,'extra_seed_trainings':12,'repeat_trainings':1,'independent_test_documents':96,'total_test_predictions':sum(r['test_documents'] for r in metrics)+repeat['test_documents']})
    print('COMPLETE',round(time.perf_counter()-start,2),'seconds',flush=True)
if __name__=='__main__':main()

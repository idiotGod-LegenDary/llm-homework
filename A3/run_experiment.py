"""Reproducible People's Daily / KenLM experiment.
Uses manually segmented tokens from a locally supplied corpus.
"""
from __future__ import annotations
import argparse, collections, gc, hashlib, json, math, platform, re, statistics
import subprocess, sys, time, os, shutil
from datetime import date
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'data'/'raw'/'199801.txt'
DIRS={n:ROOT/n for n in ['data','models','logs','results','temp']}
BIN=Path(os.environ.get('KENLM_BIN_DIR',str(ROOT/'tools'/'bin')))
ID_RE=re.compile(r'^(\d{8}-\d+-\d+)-\d+/[A-Za-z]+$')
TAG_RE=re.compile(r'^[A-Za-z]+(?:\][A-Za-z]*)*$')
PROMPT='在阳光明媚的五月，我们学校胜利召开了'
SEED=20261008

def kenlm_command(name):
    executable=name+'.exe' if os.name=='nt' else name
    local=BIN/executable
    if local.is_file():return str(local)
    installed=shutil.which(executable)
    if installed:return installed
    raise FileNotFoundError(f'{executable} not found; run setup_kenlm.py on Windows, or set --kenlm-bin / KENLM_BIN_DIR')


def save_json(path, data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def clean_line(line):
    items=line.split()
    if not items:return None
    match=ID_RE.fullmatch(items[0])
    if not match:raise ValueError('Missing document/paragraph ID: '+items[0])
    words=[]; removed_tags=0; opened=closed=0; removed_extra_markers=0
    for item in items[1:]:
        # January source: 51 word/l/% annotation artifacts, retain lexical word.
        if item.endswith("/l/%"):
            item=item[:-2];removed_extra_markers+=1
        if '/' not in item:raise ValueError('Untagged token: '+item)
        word,tag=item.rsplit('/',1)
        if not TAG_RE.fullmatch(tag):raise ValueError('Unexpected tag: '+item)
        removed_tags+=1
        # Preserve literal [/w and ]/w punctuation; remove annotation wrappers.
        if word.startswith('[') and word!='[':
            count=len(word)-len(word.lstrip('['));opened+=count;word=word[count:]
        closed+=tag.count(']')
        if not word:raise ValueError('Empty lexical token: '+item)
        if word in {'<s>','</s>','<unk>'}:raise ValueError('Reserved LM symbol in corpus')
        words.append(word)
    if opened!=closed:raise ValueError('Unbalanced entity annotation')
    return dict(doc_id=match.group(1),paragraph_id=items[0].rsplit('/',1)[0],
                words=words,raw_tokens=items[1:],removed_tags=removed_tags,
                removed_entity_wrappers=opened,removed_extra_markers=removed_extra_markers)

def check_cleaner():
    prefix='19980101-01-001-001/m '
    r=clean_line(prefix+'[中央/n 人民/n 广播/vn 电台/n]nt ，/w')
    assert r['words']==['中央','人民','广播','电台','，']
    assert r['doc_id']=='19980101-01-001'
    assert clean_line(prefix+'[/w ]/w //w')['words']==['[',']','/']
    assert clean_line(prefix+'http://example.com/n 摄/Vg')['words']==['http://example.com','摄']
    assert clean_line(prefix+'腾空而起/l/%')['words']==['腾空而起']
    try:clean_line(prefix+'[中央/n 电台/n')
    except ValueError:pass
    else:raise AssertionError('Unbalanced annotation must not be silently removed')

def prepare():
    check_cleaner()
    text=SOURCE.read_text(encoding='utf-8-sig')
    records=[]; errors=[]; blank=0
    for lineno,line in enumerate(text.splitlines(),1):
        if not line.strip():blank+=1;continue
        try:
            record=clean_line(line)
            if record and record['words']:record['source_line']=lineno;records.append(record)
        except ValueError as e:errors.append(dict(line=lineno,error=str(e),sample=line[:250]))
    save_json(DIRS['results']/'parse_errors.json',errors)
    if errors:raise ValueError(f'{len(errors)} corpus lines need inspection; see parse_errors.json')
    assigned={}
    for r in records:
        if r['doc_id'] not in assigned:
            bucket=int(hashlib.sha256(f'{SEED}:{r["doc_id"]}'.encode()).hexdigest()[:8],16)%100
            assigned[r['doc_id']]='train' if bucket<80 else 'dev' if bucket<90 else 'test'
    parts={name:[r for r in records if assigned[r['doc_id']]==name] for name in ['train','dev','test']}
    # Keep the train set authoritative; remove exact paragraph repeats inside
    # each split and those already present in earlier splits.
    seen=set(); dropped={}; cleaned={}
    for name, group in parts.items():
        kept=[]; repeats=0
        for r in group:
            key=tuple(r['words'])
            if key in seen:repeats+=1;continue
            seen.add(key);kept.append(r)
        cleaned[name]=kept;dropped[name]=repeats
        (DIRS['data']/f'{name}.txt').write_text(''.join(' '.join(r['words'])+'\n' for r in kept),encoding='utf8')
        with (DIRS['data']/f'{name}_index.jsonl').open('w',encoding='utf8') as f:
            for r in kept:f.write(json.dumps({k:r[k] for k in ['doc_id','paragraph_id','source_line']})+'\n')
    docs={name:{r['doc_id'] for r in group} for name,group in cleaned.items()}
    assert not (docs['train']&docs['dev'] or docs['train']&docs['test'] or docs['dev']&docs['test'])
    lexical={w for r in cleaned['train'] for w in r['words']}
    raw_vocab={w for r in cleaned['train'] for w in r['raw_tokens']}
    stats=dict(source=SOURCE.name,sha256=sha256(SOURCE),encoding='utf-8-sig',month='1998-01',
               input_bytes=SOURCE.stat().st_size,raw_lines=len(text.splitlines()),blank_lines=blank,
               nonempty_paragraphs=len(records),articles=len(assigned),split_seed=SEED,
               split_rule='SHA256(seed:article_id) modulo 100, train<80/dev<90/test otherwise',
               removed_document_ids=len(records),removed_pos_tags=sum(r['removed_tags'] for r in records),
               removed_entity_wrappers=sum(r['removed_entity_wrappers'] for r in records),
               removed_extra_markers=sum(r['removed_extra_markers'] for r in records),
               train_lexical_types=len(lexical),train_annotated_types=len(raw_vocab),
               raw_tagged_training_tokens=sum(len(r['raw_tokens']) for r in cleaned['train']),
               cleaner_checks='passed: compound entity, literal brackets/slash, URL, uppercase tag, extra /%, malformed entity',
               duplicates_removed=dropped, splits={})
    for name, group in cleaned.items():
        stats['splits'][name]=dict(articles=len(docs[name]),paragraphs=len(group),
              tokens=sum(len(r['words']) for r in group),bytes=(DIRS['data']/f'{name}.txt').stat().st_size,
              oov_tokens=sum(w not in lexical for r in group for w in r['words']),
              sha256=sha256(DIRS['data']/f'{name}.txt'))
    stats['examples']=[dict(raw=' '.join(r['raw_tokens']),clean=' '.join(r['words']),
                            paragraph_id=r['paragraph_id']) for r in records[:2]]
    save_json(DIRS['results']/'data_manifest.json',stats)
    print('PREPARED',json.dumps(stats['splits']),flush=True)

def train(repeats):
    runs=[]
    for rep in range(1,repeats+1):
        order_schedule=([2,3,5][(rep-1)%3:]+[2,3,5][:(rep-1)%3])
        for n in order_schedule:
            path=DIRS['models']/f'{n}gram.arpa'
            cmd=[kenlm_command('lmplz'),'-o',str(n),'-S','256M','-T',str(DIRS['temp'])+os.sep]
            start=time.perf_counter()
            with (DIRS['data']/'train.txt').open('rb') as inp,path.open('wb') as output:
                result=subprocess.run(cmd,stdin=inp,stdout=output,stderr=subprocess.PIPE)
            elapsed=time.perf_counter()-start
            (DIRS['logs']/f'lmplz_{n}gram_run{rep}.log').write_bytes(result.stderr)
            if result.returncode:raise RuntimeError(f'lmplz failed n={n}: '+result.stderr.decode('utf8',errors='replace'))
            counts={}
            with path.open(encoding='utf8') as f:
                for line in f:
                    match=re.match(r'ngram (\d+)=(\d+)',line)
                    if match:counts[match.group(1)]=int(match.group(2))
                    if line.startswith('\\1-grams:'):break
            row=dict(order=n,repeat=rep,wall_seconds=elapsed,arpa_bytes=path.stat().st_size,
                     arpa_sha256=sha256(path),ngram_counts=counts,command=['lmplz','-o',str(n),'-S','256M','-T','temp/'],
                     discount_fallback_used=False)
            runs.append(row)
            save_json(DIRS['results']/'training_runs.json',runs)
            print(f'TRAIN n={n} repeat={rep} seconds={elapsed:.4f}',flush=True)
    for n in [2,3,5]:
        hashes={r['arpa_sha256'] for r in runs if r['order']==n}
        if len(hashes)!=1:raise AssertionError('Repeated builds did not produce identical models')
        cmd=[kenlm_command('build_binary'),'probing',str(DIRS['models']/f'{n}gram.arpa'),str(DIRS['models']/f'{n}gram.bin')]
        start=time.perf_counter();result=subprocess.run(cmd,capture_output=True)
        (DIRS['logs']/f'build_binary_{n}gram.log').write_bytes(result.stdout+result.stderr)
        if result.returncode:raise RuntimeError('build_binary failed: '+result.stderr.decode('utf8',errors='replace'))
        print(f'BINARY n={n} seconds={time.perf_counter()-start:.4f}',flush=True)
    env=dict(python=sys.version,platform=platform.platform(),processor=platform.processor(),
             numpy=np.__version__,logical_cpus=__import__('os').cpu_count(),date=date.today().isoformat(),
             memory_limit='256 MiB sorting RAM',replicates=repeats,
             note='顺序轮换，重复测量不等于独立机器；时间含 lmplz 读语料和写 ARPA，排除下载、清洗、二进制转换及文件哈希。')
    save_json(DIRS['results']/'environment.json',env)

class ArpaModel:
    def __init__(self,path):
        self.words=[];self.ids={};self.base=[];self.next={};self.backoff={};self.order=0
        section=0
        with path.open(encoding='utf8') as f:
            for line in f:
                line=line.strip()
                match=re.match(r'\\(\d+)-grams:',line)
                if match:section=int(match.group(1));self.order=max(self.order,section);continue
                if not section or not line or line.startswith('\\'):continue
                fields=line.split('\t')
                if len(fields)<2:continue
                gram=fields[1].split();p=10.0**float(fields[0])
                if section==1:
                    idx=len(self.words);self.words.append(gram[0]);self.ids[gram[0]]=idx;self.base.append(p);key=(idx,)
                else:
                    key=tuple(self.ids[w] for w in gram)
                    self.next.setdefault(key[:-1],{})[key[-1]]=p
                if len(fields)>2:self.backoff[key]=10.0**float(fields[2])
        self.base=np.asarray(self.base,dtype=np.float64)
        self.bos=self.ids['<s>'];self.eos=self.ids['</s>'];self.unk=self.ids['<unk>']
    def distribution(self,history):
        p=self.base.copy()
        for k in range(1,min(self.order-1,len(history))+1):
            context=tuple(history[-k:]);p*=self.backoff.get(context,1.0)
            following=self.next.get(context)
            if following:
                indices=list(following);p[indices]=list(following.values())
        return p
    def encode(self,tokens):return [self.ids.get(w,self.unk) for w in tokens]
    def score(self,tokens):
        history=[self.bos];total=0.0
        for idx in self.encode(tokens)+[self.eos]:
            p=self.distribution(history)[idx];total+=math.log10(p);history.append(idx)
        return total
    def tokenize(self,text):
        tokens=[];i=0
        max_word=max(map(len,self.words))
        while i<len(text):
            found=None
            for end in range(min(len(text),i+max_word),i,-1):
                if text[i:end] in self.ids:found=text[i:end];break
            if found is None:found=text[i]
            tokens.append(found);i+=len(found)
        return tokens

def evaluate():
    metrics=[]
    for n in [2,3,5]:
        row=dict(order=n,arpa_bytes=(DIRS['models']/f'{n}gram.arpa').stat().st_size,
                 binary_bytes=(DIRS['models']/f'{n}gram.bin').stat().st_size)
        for split in ['dev','test']:
            cmd=[kenlm_command('query'),'-v','summary',str(DIRS['models']/f'{n}gram.bin')]
            with (DIRS['data']/f'{split}.txt').open('rb') as inp:result=subprocess.run(cmd,stdin=inp,capture_output=True)
            output=result.stdout.decode('utf8',errors='replace')
            (DIRS['logs']/f'query_{n}gram_{split}.log').write_bytes(result.stdout+result.stderr)
            if result.returncode:raise RuntimeError('query failed')
            def number(label,convert=float):
                m=re.search(r'^'+re.escape(label)+r'\s*:\s*([\d.eE+\-]+)',output,re.MULTILINE)
                if not m:raise ValueError('Cannot parse query output '+repr(output))
                return convert(m.group(1))
            row[split]=dict(ppl_including_oov=number('Perplexity including OOVs'),
                            ppl_excluding_oov=number('Perplexity excluding OOVs'),
                            oovs=number('OOVs',int),tokens=number('Tokens',int))
        metrics.append(row);print('EVAL',json.dumps(row),flush=True)
    save_json(DIRS['results']/'evaluation.json',metrics)
    return metrics

def generate():
    generated=[];checks=[]
    configs=[dict(name='greedy',mode='greedy',temperature=1.0,seeds=[42]),
             dict(name='top-k 50, T=0.7',mode='topk',k=50,temperature=0.7,seeds=[42,43,44]),
             dict(name='top-k 50, T=1.2',mode='topk',k=50,temperature=1.2,seeds=[42,43,44]),
             dict(name='top-p 0.9, T=1.0',mode='topp',p=0.9,temperature=1.0,seeds=[42,43,44])]
    examples=[line.split() for line in (DIRS['data']/'dev.txt').read_text(encoding='utf8').splitlines()[:5]]
    for n in [2,3,5]:
        model=ArpaModel(DIRS['models']/f'{n}gram.arpa');prompt_tokens=model.tokenize(PROMPT)
        # Compare the Python ARPA scorer against the actual KenLM query tool.
        # Windows build expects regular-file stdin; pipe EOF raises error 109.
        check_input=DIRS['data']/'scorer_check.txt'
        check_input.write_text('\n'.join(' '.join(x) for x in examples)+'\n',encoding='utf8')
        with check_input.open('rb') as inp:
            result=subprocess.run([kenlm_command('query'),str(DIRS['models']/f'{n}gram.bin')],
                                  stdin=inp,capture_output=True,check=True)
        (DIRS['logs']/f'python_arpa_check_{n}gram.log').write_bytes(result.stdout+result.stderr)
        totals=[float(x) for x in re.findall(r'Total:\s*([\d.eE+\-]+)',result.stdout.decode('utf8',errors='replace'))]
        if len(totals)!=len(examples):raise AssertionError('Unexpected sentence score output')
        errors=[abs(model.score(x)-y) for x,y in zip(examples,totals)]
        if max(errors)>0.01:raise AssertionError('ARPA reader disagrees with KenLM: '+str(errors))
        checks.append(dict(order=n,sentences_checked=len(examples),max_log10_score_error=max(errors),passed=True))
        for cfg in configs:
            for seed in cfg['seeds']:
                rng=np.random.default_rng(seed);history=[model.bos]+model.encode(prompt_tokens);tokens=[];stop='max_tokens'
                for step in range(48):
                    p=model.distribution(history)
                    p[model.bos]=0.0;p[model.unk]=0.0
                    if cfg['mode']=='greedy':idx=int(np.argmax(p))
                    else:
                        weights=np.power(p,1/cfg['temperature']);rank=np.argsort(weights)[::-1]
                        if cfg['mode']=='topk':eligible=rank[:cfg['k']]
                        else:
                            cumulative=np.cumsum(weights[rank])/weights.sum()
                            eligible=rank[:np.searchsorted(cumulative,cfg['p'])+1]
                        probabilities=weights[eligible];probabilities/=probabilities.sum()
                        idx=int(rng.choice(eligible,p=probabilities))
                    if idx==model.eos:stop='EOS';break
                    tokens.append(model.words[idx]);history.append(idx)
                generated.append(dict(order=n,config=cfg['name'],seed=seed,prompt=PROMPT,
                   prompt_tokens=prompt_tokens,prompt_oov=sum(w not in model.ids for w in prompt_tokens),
                   continuation_tokens=tokens,text=PROMPT+''.join(tokens),stop_reason=stop,
                   generated_tokens=len(tokens),repetition_ratio=1-len(set(tokens))/len(tokens) if tokens else 0))
        print(f'GENERATED n={n}, ARPA score cross-check max_error={max(errors):.7f}',flush=True)
        del model;gc.collect()
    save_json(DIRS['results']/'generation.json',generated)
    save_json(DIRS['results']/'arpa_scorer_checks.json',checks)
    md=['# A3 续写样本\n','模型直接输出；省略号为待续写标记，不作为输入 token。\n']
    for row in generated:
        md += [f"## {row['order']}-gram / {row['config']} / seed={row['seed']}\n",
               f"分词提示：`{' '.join(row['prompt_tokens'])}`。停止原因：{row['stop_reason']}。\n",row['text']+'\n']
    (DIRS['results']/'续写样本.md').write_text('\n'.join(md),encoding='utf8')

def summary():
    data=json.loads((DIRS['results']/'data_manifest.json').read_text(encoding='utf8'))
    runs=json.loads((DIRS['results']/'training_runs.json').read_text(encoding='utf8'))
    metrics=json.loads((DIRS['results']/'evaluation.json').read_text(encoding='utf8'))
    md=['# A3 实验结果\n','## 数据\n','| 集合 | 文章 | 段落 | 词 token | OOV token |','|---|---:|---:|---:|---:|']
    for name,row in data['splits'].items():md.append(f"| {name} | {row['articles']} | {row['paragraphs']} | {row['tokens']} | {row['oov_tokens']} |")
    md += ['\n## 阶数、训练时间与测试指标\n','重复次数见 environment.json。时间为 wall time；PPL 包含 OOV 和各段落的 EOS。\n',
       '| n | 三次训练时间（秒） | 平均 ± 标准差（秒） | ARPA MiB | 二进制 MiB | 验证 PPL | 测试 PPL |','|---:|---|---|---:|---:|---:|---:|']
    for row in metrics:
        times=[r['wall_seconds'] for r in runs if r['order']==row['order']]
        md.append(f"| {row['order']} | {', '.join(f'{x:.3f}' for x in times)} | {statistics.mean(times):.3f} ± {statistics.stdev(times) if len(times)>1 else 0:.3f} | {row['arpa_bytes']/2**20:.2f} | {row['binary_bytes']/2**20:.2f} | {row['dev']['ppl_including_oov']:.3f} | {row['test']['ppl_including_oov']:.3f} |")
    (DIRS['results']/'实验结果.md').write_text('\n'.join(md)+'\n',encoding='utf8')
    print('\n'.join(md),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description="People's Daily n-gram training and generation")
    parser.add_argument('--stage',choices=['all','prepare','train','analyze'],default='all')
    parser.add_argument('--repeats',type=int,choices=range(1,11),default=3)
    parser.add_argument('--source',type=Path,default=SOURCE,help='Path to the annotated January corpus')
    parser.add_argument('--kenlm-bin',type=Path,default=BIN,help='Directory containing lmplz, build_binary and query')
    parser.add_argument('--output-dir',type=Path,default=ROOT,help='Root directory for generated data/models/logs/results')
    args=parser.parse_args()
    SOURCE=args.source.expanduser().resolve();BIN=args.kenlm_bin.expanduser().resolve()
    DIRS={name:args.output_dir.resolve()/name for name in ['data','models','logs','results','temp']}
    for directory in DIRS.values():directory.mkdir(parents=True,exist_ok=True)
    if args.stage in ['all','prepare'] and not SOURCE.is_file():
        parser.error('Corpus not found; supply --source /path/to/199801.txt')
    if args.stage in ['all','train','analyze']:
        for tool in ['lmplz','build_binary','query']:kenlm_command(tool)
    if args.stage in ['all','prepare']:prepare()
    if args.stage in ['all','train']:train(args.repeats)
    if args.stage in ['all','analyze']:evaluate();generate();summary()

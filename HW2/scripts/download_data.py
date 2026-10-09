"""Download fixed prefixes of eight category files from a pinned THUCNews mirror."""
import argparse, concurrent.futures, hashlib, json, time, urllib.request, urllib.parse
from pathlib import Path
REVISION = 'ae77e363396e90b54cdaa9eafdbc5e102a3d4019'
LABELS = ['体育','财经','教育','娱乐','科技','房产','游戏','时政']
ROOT = Path(__file__).resolve().parents[1]
def sha(data): return hashlib.sha256(data).hexdigest()
def fetch(label, count, folder):
    path = folder / (label + '.jsonl')
    meta_path = folder / (label + '.meta.json')
    if path.exists() and meta_path.exists():
        meta=json.loads(meta_path.read_text(encoding='utf8'))
        if meta['sha256']==sha(path.read_bytes()) and meta['records']==count:
            return meta
    url='https://huggingface.co/datasets/Tongjilibo/THUCNews/resolve/'+REVISION+'/'+urllib.parse.quote(label+'.jsonl')
    for attempt in range(4):
        try:
            req=urllib.request.Request(url,headers={'Range':'bytes=0-14999999','User-Agent':'llm-homework-data/1.0'})
            with urllib.request.urlopen(req,timeout=60) as response:
                chunks=[]
                for i in range(count):
                    line=response.readline()
                    if not line.endswith(b'\n'):
                        raise RuntimeError('Requested byte prefix does not contain enough complete records: '+label)
                    row=json.loads(line)
                    assert {'id','title','content'}<=set(row)
                    chunks.append(line)
                data=b''.join(chunks)
                meta={'label':label,'revision':REVISION,'url':url,'records':count,'bytes_saved':len(data),'sha256':sha(data),'etag':response.headers.get('ETag'),'content_range':response.headers.get('Content-Range'),'requested_range':'bytes=0-14999999','saved_range':f'bytes=0-{len(data)-1}','selection':'first complete records in category file; not population-random'}
            path.write_bytes(data)
            meta_path.write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf8')
            return meta
        except Exception:
            if attempt==3: raise
            time.sleep(2**attempt)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--per-class',type=int,default=1250);args=parser.parse_args()
    folder=ROOT/'data'/'raw';folder.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();metadata=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        tasks={pool.submit(fetch,label,args.per_class,folder):label for label in LABELS}
        for task in concurrent.futures.as_completed(tasks):
            meta=task.result();metadata.append(meta)
            print('DOWNLOADED',meta['label'],meta['records'],meta['bytes_saved'],flush=True)
    metadata.sort(key=lambda x:LABELS.index(x['label']))
    results=ROOT/'results';results.mkdir(exist_ok=True)
    (results/'download_manifest.json').write_text(json.dumps({'revision':REVISION,'per_class':args.per_class,'seconds':time.perf_counter()-start,'files':metadata},ensure_ascii=False,indent=2),encoding='utf8')
    print('FINISHED',sum(x['records'] for x in metadata),'records',flush=True)
if __name__=='__main__': main()

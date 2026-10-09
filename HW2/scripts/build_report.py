"""Build Markdown tables from completed local results, without inserting sample scores."""
import csv,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results'
def load(name):return json.loads((R/name).read_text(encoding='utf8'))
with (R/'classification_metrics.csv').open(encoding='utf-8-sig') as f:metrics=list(csv.DictReader(f))
for row in metrics:
    for k in ['classes','seed','correct','train_documents','test_documents','classifier_vocabulary']:row[k]=int(row[k])
    for k in ['accuracy','macro_f1','train_seconds','predict_seconds']:row[k]=float(row[k])
data=load('data_statistics.json');emb=load('embedding_training.json');ana=load('embedding_analysis.json');run=load('run_summary.json');verify=load('verification.json');repeat=load('classification_repeat.json')
labels=load('classifier_config.json')['labels']
train_words=set()
for line in (ROOT/'data/processed/8_train.txt').read_text(encoding='utf8').splitlines():train_words.update(line.split()[1:])
pretrained_words=set()
with (ROOT/'models/fasttext_pretrained.vec').open(encoding='utf8') as f:
    header=f.readline()
    for line in f:pretrained_words.add(line.split(' ',1)[0])
test_tokens=[]
for line in (ROOT/'data/processed/8_test.txt').read_text(encoding='utf8').splitlines():test_tokens.extend(line.split()[1:])
coverage={'test_token_count':len(test_tokens),'supervised_train_token_coverage':sum(w in train_words for w in test_tokens)/len(test_tokens),'pretrained_vocabulary_token_coverage':sum(w in pretrained_words for w in test_tokens)/len(test_tokens),'vec_header':header.strip(),'supervised_train_unique_tokens':len(train_words),'pretrained_unique_words':len(pretrained_words)}
(R/'token_coverage.json').write_text(json.dumps(coverage,ensure_ascii=False,indent=2),encoding='utf8')
lines=['# 用 Gensim 跑表示学习与 fastText 文本分类','',
'本次实际完成了两个学习阶段：先从未使用测试文章的新闻语料训练 Word2Vec 和 FastText 词向量，再用带标签新闻训练官方 fastText 监督分类器，比较导入 FastText 词向量与从零训练的效果。本文记录数据、参数、实测结果和失败分析。','',
'## 一、数据与实验设计','',
f'数据取自 [THUCNews 项目](https://github.com/thunlp/THUCTC) 的 [固定镜像版本](https://huggingface.co/datasets/Tongjilibo/THUCNews/tree/ae77e363396e90b54cdaa9eafdbc5e102a3d4019)。选择八个类别文件各自前 1,250 条，共 **{data["raw_documents"]:,} 篇**。这是固定文件前缀采样，不能视为从整个新闻总体随机抽样；镜像与原始分发也没有逐篇核验一致性。下载 URL、实际保存前缀长度和 SHA-256 均已记录。','',
f'去除空白后不足 100 字、完整正文重复、前 200 字重复，以及分词后不足 50 词的文章后，保留 **{data["retained_documents"]:,} 篇**。具体排除数：短／空正文 {data["rejected"].get("short_or_empty_body",0)} 篇，完整重复 {data["rejected"].get("duplicate_body",0)} 篇，前 200 字重复 {data["rejected"].get("duplicate_first_200_chars",0)} 篇，分词后过短 {data["rejected"].get("fewer_than_50_tokens",0)} 篇。正文分句后用 jieba（HMM=False）分词，删除标点 token，丢弃不足 3 词的片段。','',
f'类别直接继承源文件名，不根据标题生成标签，也没有声称逐篇人工重新标注。每类按种子 42 与文章 ID 的 SHA-256 排序，选 **48 篇监督训练、12 篇测试**。分类输入只用正文的前 500 个词；标题只在错误分析中用于定位。全部 **96 篇测试文章**都先排除，再训练词向量，得到 **{data["representation_documents"]:,} 篇、{data["representation_sentences"]:,} 句、{data["representation_tokens"]:,} 词次**的表示学习语料，其中有 {data["unlabeled_documents"]:,} 篇额外未标注材料。','',
'| 类别数 | 类别 | 监督训练 / 测试 |','| --- | --- | --- |']
for n in [2,4,8]:lines.append(f'| {n} | {"、".join(labels[:n])} | {n*48} / {n*12} |')
lines+=['','2／4／8 类实验沿用相同文章的嵌套划分，不重新挑选容易的测试样本。同一文章及相同正文／前 200 字不会跨训练和测试两侧，但这些检查不能排除所有同事件改写。本次分类语料与题目示例的人民日报语料不同，八类也不同，不能直接比较两份作业的分数。','',
'## 二、表示学习：Word2Vec 与 Gensim FastText','',
'Word2Vec 的 skip-gram 从中心词和相邻词的共现学习表示；Gensim FastText 还为词内字符片段学习表示，组合得到词向量。因此相近向量是语料统计的结果，不能直接当成词典意义。表示模型没有类别输出，后面还需要带标签训练分类映射。','',
'共同参数：skip-gram、100 维、窗口 5、最低词频 3、5 轮、单线程、种子 42，启动 Python 前设置 `PYTHONHASHSEED=0`。FastText 另外使用长度 2—4 的字符 n-gram、200,000 个哈希桶；其余采用所记录版本的默认参数。两者逐行读取同一个分词语料文件。[Word2Vec 文档](https://radimrehurek.com/gensim/models/word2vec.html) · [FastText 文档](https://radimrehurek.com/gensim/models/fasttext.html)','',
'| 模型 | 词表规模 | 本次构建与训练耗时（秒） |','| --- | ---: | ---: |']
for name in ['Word2Vec','FastText']:lines.append(f'| {name} | {emb[name]["vocabulary"]:,} | {emb[name]["seconds"]:.2f} |')
lines+=['','这是同一台机器的一次运行耗时，包含读取、建词表和训练，不是普遍速度排名。','',
'### 1. 近邻观察','',
'| 查询词 | Word2Vec 前 3 个近邻 | FastText 前 3 个近邻 |','| --- | --- | --- |']
for word in ['银行','学校','电脑','经济']:
    neighbors=[]
    for name in ['Word2Vec','FastText']:neighbors.append('、'.join(n['word'] for n in ana[name]['neighbors'].get(word,[])[:3]))
    lines.append(f'| {word} | {neighbors[0]} | {neighbors[1]} |')
lines+=['','这里只展示四个查询；全部前 10 个近邻及余弦值已保存。共享汉字可能影响 FastText 近邻，需要结合含义检查，不能仅按是否共享字判断好坏。','',
'### 2. 词表外查询','',
'查询“量子计算芯片”“智能养老机器人”“跨境数字人民币钱包”，用 `key_to_index` 检查是否真正进入训练词表，再尝试取得向量。','',
'| 查询 | Word2Vec：在词表 / 可返回向量 | FastText：在词表 / 可返回向量 |','| --- | --- | --- |']
for word in ana['FastText']['oov']:
    values=[]
    for name in ['Word2Vec','FastText']:
        row=ana[name]['oov'][word];values.append(('是' if row['in_training_vocabulary'] else '否')+' / '+('是' if row['vector_available'] else '否'))
    lines.append(f'| {word} | {values[0]} | {values[1]} |')
lines+=['','Gensim FastText 利用字符片段合成词表外向量，能返回数值不等于理解了新词；本次还保留了向量范数与返回近邻，没有把“接口成功”计为语义正确。[子词表示论文](https://aclanthology.org/Q17-1010/)','',
'### 3. 四组词的相似度','',
'预先固定体育（足球、篮球、比赛、运动员）、财经（银行、贷款、股票、利率）、教育（学校、学生、教师、课程）、科技（电脑、软件、网络、芯片）四组词。计算无序词对的组内与跨组余弦均值，并记录缺词情况。','',
'| 模型 | 组内均值 | 跨组均值 | 差值 | 有效组内 / 跨组词对 |','| --- | ---: | ---: | ---: | --- |']
for name in ['Word2Vec','FastText']:
    g=ana[name]['group_similarity'];lines.append(f'| {name} | {g["within_mean"]:.4f} | {g["cross_mean"]:.4f} | {g["gap"]:.4f} | {g["within_pairs"]} / {g["cross_pairs"]} |')
lines+=['','这 16 个词均在两个模型的训练词表中。这些词是很小的定性探针，两项平均值都高也不能推出语义更好，更不能替代下游分类评测。','',
'## 三、监督分类：导入词向量与从零训练','',
'将 Gensim FastText 的词表内词向量导出为文本 `.vec`，显式传入 **另一套库**的 `fasttext.train_supervised(pretrainedVectors=...)`。`.vec` 没有转移完整的字符片段参数；Gensim 词向量训练本身也不会自动得到本文的监督分类器。[fastText Python 接口](https://fasttext.cc/docs/en/python-module.html)','',
'分类参数为 100 维、学习率 0.3、25 轮、词二元组、softmax、单线程、最低词频 1、200,000 个哈希桶。字符片段参数 `minn=maxn=0`，因此这一阶段不启用字符子词；词二元组特征仍保留。对照仅去掉 `pretrainedVectors`，沿用同一输入文件、参数与划分。Windows 使用 `fasttext-wheel==0.9.2` 的第三方预编译分发，执行的是 fastText 监督接口。[监督分类论文](https://aclanthology.org/E17-2068/) · [分发记录](https://pypi.org/project/fasttext-wheel/0.9.2/)','',
'### 1. 主结果：随机种子 42','',
'| 类别数 | 导入词向量：答对 / 测试；准确率 / 宏 F1 | 从零训练：答对 / 测试；准确率 / 宏 F1 |','| --- | --- | --- |']
for n in [2,4,8]:
    cells=[]
    for init in ['pretrained','scratch']:
        row=next(r for r in metrics if r['classes']==n and r['initialization']==init and r['seed']==42)
        cells.append(f'{row["correct"]}/{row["test_documents"]}；{row["accuracy"]:.3f} / {row["macro_f1"]:.3f}')
    lines.append(f'| {n} | {cells[0]} | {cells[1]} |')
lines+=['','![分类结果](results/classification_scores.png)','',
'### 2. 分类随机性的检查','',
'额外用种子 43、44 跑相同六个条件。下面为三次分类训练的均值 ± 样本标准差；测试集和词向量均没有改变。','',
'| 类别数 | 初始化 | 准确率均值 ± 标准差 | 宏 F1 均值 ± 标准差 |','| --- | --- | --- | --- |']
for n in [2,4,8]:
    for init,name in [('pretrained','导入词向量'),('scratch','从零训练')]:
        rows=[r for r in metrics if r['classes']==n and r['initialization']==init]
        a=[r['accuracy'] for r in rows];f=[r['macro_f1'] for r in rows]
        lines.append(f'| {n} | {name} | {statistics.mean(a):.3f} ± {statistics.stdev(a):.3f} | {statistics.mean(f):.3f} ± {statistics.stdev(f):.3f} |')
lines+=['',f'另外，8 类导入词向量模型以种子 42 重训一次，{repeat["matching_predictions"]}/{repeat["test_documents"]} 篇预测与第一次相同。这个检查支持本次环境内的重复性；三个种子不是三个独立测试集，也没有证明换语料或换划分仍然保持优势。','',
f'在 8 类测试正文中共有 {coverage["test_token_count"]:,} 个输入词次：在监督训练词集合中出现过的占 {coverage["supervised_train_token_coverage"]:.2%}，在导出预训练词表中出现的占 {coverage["pretrained_vocabulary_token_coverage"]:.2%}。这说明额外未标注语料扩大了可复用的词表示覆盖，但覆盖率本身不是分类正确率。','',
'### 3. 补充收敛诊断','',
'看到 25 轮从零训练的低分后，补充 8 类、种子 42 的 100 轮和 250 轮实验，其他参数与文章不变。25 轮两个模型由已保存的二进制重新加载并评估训练集／测试集，没有重复训练它们。','',
'| 初始化 / 轮数 | 训练准确率 | 测试答对 / 96 | 测试准确率 / 宏 F1 | 新训练耗时（秒） |','| --- | ---: | ---: | --- | ---: |']
with (R/'convergence_metrics.csv').open(encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        name='导入词向量' if row['initialization']=='pretrained' else '从零训练'
        seconds=f"{float(row['train_seconds']):.2f}" if row['new_training']=='True' else '加载已有模型'
        lines.append(f"| {name} / {row['epoch']} | {float(row['train_accuracy']):.3f} | {row['test_correct']} | {float(row['test_accuracy']):.3f} / {float(row['test_macro_f1']):.3f} | {seconds} |")
lines+=['','从零训练 25 轮的训练准确率也只有 39.6%，支持训练不足的诊断；100／250 轮训练准确率均为 100%，测试准确率升到 81.3%／86.5%。因此 25 轮的大分差不能解释成从零训练最终只能达到 32.3%。补充实验是在观察主结果后开展的，复用了同一测试集，仅用于诊断；没有把其中最高分当成独立验证过的调参结果。','',
'### 4. 错误分析','',
'![8 类混淆矩阵](results/classification_confusion_8.png)','',
(R/'error_analysis.md').read_text(encoding='utf8'),'',
'## 四、实际完成的工作与验证','',
f'- 数据下载：8 个固定版本文件前缀，{data["raw_documents"]:,} 篇原始新闻；保存来源、长度、ETag 和 SHA-256。',
f'- 表示学习：实际训练 Word2Vec、FastText 各 1 个；查询 4 个词的近邻、3 个词表外组合，并计算固定四组词的组内／跨组词对。',
'- 监督分类：2／4／8 类 × 2 种初始化 × 3 个随机种子，共 18 次训练，再进行 1 次相同种子重复，主流程共 **19 次分类训练**；收敛诊断另外训练 100／250 轮模型各 1 个，合计 **21 次分类训练**。',
f'- 逐条预测：主对照和额外种子共 {verify["prediction_rows"]:,} 条，重复检查 96 条，共 **{run["total_test_predictions"]:,} 次预测**；收敛诊断另有 384 次测试预测和 1,536 次训练集预测（评估 4 个条件，其中 2 个为加载已有模型）；两部分共 1,488 次测试预测，仍来自 **96 篇独立测试文章**，没有把重复预测当作独立样本。',
'- 验证：检查文章 ID、正文哈希、前缀哈希唯一，测试集排除于词向量语料，监督训练／测试隔离及嵌套划分；从逐条预测独立复算 18 行准确率、宏 F1 和混淆矩阵，结果一致；另从保存预测复算 4 个收敛条件的训练准确率、测试准确率、测试宏 F1 与混淆矩阵。重复运行的逐条相等在训练进程内断言，再核对保存记录。',
f'- 主流程已计时的预处理、表示模型构建训练、分类训练与预测合计约 {run["timed_phases_seconds"]:.1f} 秒；不包含下载、安装、模型存盘、绘图、失败调试和文稿整理。','',
'收敛诊断另记录 4 个模型条件的评估和 2 次新增训练耗时。完整数值、日志和逐条预测见 `results/`；输入哈希在 `data_statistics.json`，逐篇划分在 `document_manifest.csv`。复现步骤见 [REPRODUCE.md](REPRODUCE.md)，来源与局限见 [DATA.md](DATA.md)。','',
'## 五、结论边界与后续工作','',
(R/'conclusion.md').read_text(encoding='utf8'),'',
'本次使用每类 48 篇监督训练、12 篇留出测试。文件前缀采样、固定划分、跨主题报道和未排除的同事件改写都限制了结论。三个分类种子只检查训练随机性；没有评测领域外新闻，也没有训练 Doc2Vec 或系统调整数据规模。这些可以作为后续扩展，不能算入已完成结果。','',
'公开代码与报告：[llm-homework / HW2](https://github.com/idiotGod-LegenDary/llm-homework/tree/main/HW2)。仓库包含脚本、依赖、报告、图和结果，完整语料及模型由本地脚本生成。尚未提交 Piazza。','']
text='\n'.join(lines)
(ROOT/'README.md').write_text(text,encoding='utf8');(ROOT/'Piazza提交稿.md').write_text(text,encoding='utf8')
print('Report tables generated from completed results')

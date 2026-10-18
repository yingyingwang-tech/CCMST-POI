import argparse,csv,json,random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from ccmst.data import prepare,synthetic,MobilityDataset
from ccmst.model import CCMSTPOI,objective
from ccmst.evaluation import predict,summarize,write_rows,calibration_metrics,fit_temperature,paired_cluster_bootstrap,device_batch
from ccmst.perturb import perturb

def seedall(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def build_model(meta,args):return CCMSTPOI(meta['num_poi'],meta['num_category'],d=args.dim,heads=args.heads,layers=args.layers,windows=tuple(args.windows),fusion=args.fusion).to(args.device)
def dataset_loader(data,split,batch,window,shuffle=False):
    ds=MobilityDataset(data,split,window)
    if not len(ds):raise RuntimeError(f'No eligible {split} instances (unknown POI targets excluded).')
    return DataLoader(ds,batch_size=batch,shuffle=shuffle,num_workers=0)
def train(args):
    seedall(args.seed);meta=json.loads((Path(args.data)/'meta.json').read_text());model=build_model(meta,args)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    tr=dataset_loader(args.data,'train',args.batch,max(args.windows),True)
    va=dataset_loader(args.data,'val',args.batch,max(args.windows))
    opt=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    best=-1;patience=0;history=[]
    for epoch in range(1,args.epochs+1):
        model.train();tot=0;count=0
        for b in tr:
            b=device_batch(b,args.device);opt.zero_grad();outp=model(b)
            loss,parts=objective(outp,b['label'],args.aux_weight,args.rel_weight,args.target_temperature)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            opt.step();tot+=float(loss.detach())*len(b['label']);count+=len(b['label'])
        rows=predict(model,va,args.device);metrics=summarize(rows)
        record={'epoch':epoch,'loss':tot/count,**metrics};history.append(record)
        print(json.dumps(record))
        if metrics['ndcg_10']>best:
            best=metrics['ndcg_10'];patience=0
            torch.save({'model':model.state_dict(),'meta':meta,'config':{k:getattr(args,k) for k in ['dim','heads','layers','windows','fusion']},'epoch':epoch},out/'best.pt')
        else:
            patience+=1
            if patience>=args.patience:break
    (out/'history.json').write_text(json.dumps(history,indent=2));print('Saved:',out/'best.pt')
def load_model(args):
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    cfg=checkpoint['config'];model=CCMSTPOI(checkpoint['meta']['num_poi'],checkpoint['meta']['num_category'],d=cfg['dim'],heads=cfg['heads'],layers=cfg['layers'],windows=tuple(cfg['windows']),fusion=cfg['fusion']).to(args.device)
    model.load_state_dict(checkpoint['model']);model.eval();return model,max(cfg['windows'])
def evaluate(args):
    model,window=load_model(args);loader=dataset_loader(args.data,args.split,args.batch,window)
    rows,raw=predict(model,loader,args.device,True);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    write_rows(rows,out/f'{args.split}_predictions.csv')
    np.savez_compressed(out/f'{args.split}_logits.npz',**raw)
    report=summarize(rows);report['calibration_uncalibrated']=calibration_metrics(raw['logits'],raw['labels'])
    (out/f'{args.split}_metrics.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
def calibration(args):
    val=np.load(args.val);test=np.load(args.test)
    if val['logits'].shape[1]!=test['logits'].shape[1]:raise ValueError('Candidate universe mismatch')
    T=fit_temperature(val['logits'],val['labels']);res={'before':calibration_metrics(test['logits'],test['labels']),'after':calibration_metrics(test['logits'],test['labels'],T)}
    Path(args.out).parent.mkdir(parents=True,exist_ok=True);Path(args.out).write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
def stats(args):
    def read(p):
        with open(p,newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
    a,b=read(args.model_a),read(args.model_b)
    out={m:paired_cluster_bootstrap(a,b,m,args.reps,args.seed) for m in ('recall_10','ndcg_10','rr')}
    Path(args.out).parent.mkdir(parents=True,exist_ok=True);Path(args.out).write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
def robustness(args):
    model,window=load_model(args);ds=MobilityDataset(args.data,'test',window)
    if args.max_samples: ds.items=ds.items[:args.max_samples]
    loader=DataLoader(ds,batch_size=1,shuffle=False)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);results=[]
    for mode in args.modes:
        levels=[0,.1,.2,.3,.4] if mode in ('random','block','recent') else [0,15,30,60,120] if mode=='time' else [0,50,100,200,500]
        for level in levels:
            for repeat in range(args.repeats if mode not in ('recent',) else 1):
                from ccmst.evaluation import per_instance
                vals=[]
                for idx,b in enumerate(loader):
                    bb=perturb(b,mode,level,args.seed+repeat*100000+idx)
                    with torch.no_grad():pred=model(device_batch(bb,args.device))['logits']
                    metr=per_instance(pred,bb['label']);vals.append((float(metr['recall_10'][0]),float(metr['ndcg_10'][0])))
                v=np.mean(vals,axis=0);r={'mode':mode,'level':level,'repeat':repeat,'recall_10':float(v[0]),'ndcg_10':float(v[1])}
                results.append(r);print(r)
    write_rows(results,out/'robustness.csv')
def main():
    p=argparse.ArgumentParser(description='CCMST-POI reproducible research implementation')
    sub=p.add_subparsers(dest='cmd',required=True)
    x=sub.add_parser('synthetic-data');x.add_argument('--out',required=True);x.add_argument('--users',type=int,default=20)
    x=sub.add_parser('prepare');x.add_argument('--csv',required=True);x.add_argument('--out',required=True);x.add_argument('--min-user',type=int,default=20);x.add_argument('--min-poi',type=int,default=10)
    x=sub.add_parser('train');x.add_argument('--data',required=True);x.add_argument('--out',required=True);x.add_argument('--epochs',type=int,default=100);x.add_argument('--patience',type=int,default=10);x.add_argument('--batch',type=int,default=256)
    x.add_argument('--dim',type=int,default=64);x.add_argument('--heads',type=int,default=4);x.add_argument('--layers',type=int,default=2)
    x.add_argument('--windows',type=int,nargs=3,default=[10,30,100]);x.add_argument('--fusion',choices=['reliability','quality','uniform','attention','gating'],default='reliability')
    x.add_argument('--aux-weight',type=float,default=.2);x.add_argument('--rel-weight',type=float,default=.1);x.add_argument('--target-temperature',type=float,default=1.)
    x.add_argument('--lr',type=float,default=.001);x.add_argument('--weight-decay',type=float,default=1e-5);x.add_argument('--seed',type=int,default=42);x.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    x=sub.add_parser('evaluate');x.add_argument('--data',required=True);x.add_argument('--checkpoint',required=True);x.add_argument('--split',default='test',choices=['train','val','test']);x.add_argument('--out',required=True);x.add_argument('--batch',type=int,default=128);x.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    x=sub.add_parser('calibrate');x.add_argument('--val',required=True);x.add_argument('--test',required=True);x.add_argument('--out',required=True)
    x=sub.add_parser('stats');x.add_argument('--model-a',required=True);x.add_argument('--model-b',required=True);x.add_argument('--out',required=True);x.add_argument('--reps',type=int,default=3000);x.add_argument('--seed',type=int,default=42)
    x=sub.add_parser('robustness');x.add_argument('--data',required=True);x.add_argument('--checkpoint',required=True);x.add_argument('--out',required=True);x.add_argument('--modes',nargs='+',default=['random','block','recent','time','geo']);x.add_argument('--repeats',type=int,default=5);x.add_argument('--max-samples',type=int,default=0);x.add_argument('--seed',type=int,default=42);x.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    a=p.parse_args()
    if a.cmd=='synthetic-data':synthetic(a.out,a.users);print('Generated synthetic data:',a.out)
    elif a.cmd=='prepare':print(prepare(a.csv,a.out,a.min_user,a.min_poi))
    elif a.cmd=='train':train(a)
    elif a.cmd=='evaluate':evaluate(a)
    elif a.cmd=='calibrate':calibration(a)
    elif a.cmd=='stats':stats(a)
    else:robustness(a)
if __name__=='__main__':main()

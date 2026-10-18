import csv,json,math
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F

def device_batch(batch,device):return {k:(v.to(device) if torch.is_tensor(v) else v) for k,v in batch.items()}
def per_instance(logits,labels,ks=(5,10,20)):
    ranks=(logits.argsort(dim=-1,descending=True)==labels[:,None]).int().argmax(-1)+1
    r=ranks.cpu().numpy();out={'rank':r,'rr':1/r.astype(float)}
    for k in ks:
        hit=(r<=k).astype(float);out[f'recall_{k}']=hit
        out[f'ndcg_{k}']=hit/np.log2(r+1)
    return out
@torch.no_grad()
def predict(model,loader,device,return_logits=False):
    model.eval();rows=[];logits_out=[];labels_out=[];ids=[]
    for b in loader:
        out=model(device_batch(b,device));logits=out['logits'];labels=b['label'].to(device)
        metrics=per_instance(logits,labels)
        probs=F.softmax(logits,dim=-1);conf,pred=probs.max(-1)
        for i in range(len(b['instance'])):
            row={'user_id':b['user'][i],'instance_id':b['instance'][i],'label':int(labels[i]),'pred':int(pred[i]),'top1_confidence':float(conf[i])}
            row.update({k:float(v[i]) for k,v in metrics.items()});rows.append(row)
        if return_logits:
            logits_out.append(logits.cpu().numpy().astype('float32'));labels_out.append(labels.cpu().numpy());ids.extend(b['instance'])
    if return_logits:
        return rows,{'logits':np.concatenate(logits_out),'labels':np.concatenate(labels_out),'instance_ids':np.array(ids)}
    return rows

def summarize(rows):
    keys=['recall_5','recall_10','recall_20','ndcg_5','ndcg_10','ndcg_20','rr']
    return {k:float(np.mean([r[k] for r in rows])) for k in keys}
def write_rows(rows,path):
    if not rows:return
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def calibration_metrics(logits,labels,T=1.,bins=15):
    logits=torch.as_tensor(logits,dtype=torch.float64)/T
    labels=torch.as_tensor(labels,dtype=torch.long)
    probs=F.softmax(logits,dim=1);conf,pred=probs.max(1);accuracy=(pred==labels).double()
    edges=torch.linspace(0,1,bins+1,dtype=torch.float64)
    ece=0.
    for i in range(bins):
        m=(conf>=edges[i]) & ((conf<=edges[i+1]) if i==bins-1 else (conf<edges[i+1]))
        if m.any():ece+=float(m.double().mean()*(conf[m].mean()-accuracy[m].mean()).abs())
    nll=float(F.cross_entropy(logits,labels))
    # Multi-class full candidate Brier (sum over candidate classes per sample).
    brier=float(((probs-F.one_hot(labels,probs.shape[1]))**2).sum(1).mean())
    return {'ece_top1':ece,'nll':nll,'brier_multiclass':brier,'top1_accuracy':float(accuracy.mean()),'temperature':T}
def fit_temperature(logits,labels):
    z=torch.tensor(logits,dtype=torch.float64);y=torch.tensor(labels,dtype=torch.long)
    log_t=torch.tensor(0.,dtype=torch.float64,requires_grad=True)
    optimizer=torch.optim.LBFGS([log_t],lr=.2,max_iter=80,line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad();loss=F.cross_entropy(z/torch.exp(log_t.clamp(-4,4)),y);loss.backward();return loss
    optimizer.step(closure)
    return float(log_t.detach().clamp(-4,4).exp())

def paired_cluster_bootstrap(rows_a,rows_b,metric='recall_10',reps=1000,seed=42):
    a={r['instance_id']:r for r in rows_a};b={r['instance_id']:r for r in rows_b}
    if set(a)!=set(b):raise ValueError('Prediction instance IDs differ; paired comparison invalid')
    groups=defaultdict(list)
    for key,ra in a.items():
        rb=b[key]
        if ra['user_id']!=rb['user_id']:raise ValueError('User mismatch')
        groups[ra['user_id']].append(float(ra[metric])-float(rb[metric]))
    keys=list(groups);num=np.array([len(groups[k]) for k in keys]);sums=np.array([sum(groups[k]) for k in keys])
    delta=float(sums.sum()/num.sum());rng=np.random.default_rng(seed);boots=[]
    for _ in range(reps):
        indices=rng.integers(0,len(keys),size=len(keys))
        boots.append(float(sums[indices].sum()/num[indices].sum()))
    return {'delta':delta,'ci95':[float(x) for x in np.percentile(boots,[2.5,97.5])], 'clusters':len(keys),'bootstrap_reps':reps}

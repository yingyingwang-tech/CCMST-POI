import csv,json,math,random
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

REQUIRED=['user_id','poi_id','timestamp','latitude','longitude','category']
def prepare(source,out,min_user=20,min_poi=10,split=(.7,.1,.2)):
    """Generic normalized CSV adapter. Split after filtering (paper-like protocol; see README leakage note)."""
    from datetime import datetime,timezone
    records=[]
    with open(source,encoding='utf-8-sig',newline='') as f:
        r=csv.DictReader(f)
        if not set(REQUIRED).issubset(r.fieldnames): raise ValueError(f'Required columns: {REQUIRED}')
        for row in r:
            try:
                ts=float(row['timestamp'])
                lat,lon=float(row['latitude']),float(row['longitude'])
                if not (-90<=lat<=90 and -180<=lon<=180):continue
                records.append({'u':row['user_id'],'v':row['poi_id'],'t':ts,'lat':lat,'lon':lon,'c':str(row['category'])})
            except (ValueError,TypeError,KeyError):continue
    # Iteratively remove under-observed users/POIs.
    while True:
        uc=Counter(r['u'] for r in records);vc=Counter(r['v'] for r in records)
        fresh=[r for r in records if uc[r['u']]>=min_user and vc[r['v']]>=min_poi]
        if len(fresh)==len(records):break
        records=fresh
    users=defaultdict(list)
    for row in records:users[row['u']].append(row)
    for u in users:
        users[u].sort(key=lambda r:(r['t'],r['v']))
        users[u]=[r for i,r in enumerate(users[u]) if i==0 or (r['v'],r['t'],r['lat'],r['lon'])!=(users[u][i-1]['v'],users[u][i-1]['t'],users[u][i-1]['lat'],users[u][i-1]['lon'])]
    pv={'<PAD>':0,'<UNK>':1}; cv={'<PAD>':0,'<UNK>':1}
    # Registry is built on TRAIN TARGET/HISTORY interactions only; unknown validation/test POIs are separately counted.
    for seq in users.values():
        n=len(seq);a=max(1,int(n*split[0]))
        for r in seq[:a]:
            if r['v'] not in pv:pv[r['v']]=len(pv)
            if r['c'] not in cv:cv[r['c']]=len(cv)
    outputs=[]; unseen=0
    from datetime import datetime,timezone
    for u,seq in users.items():
        n=len(seq);a=max(1,int(n*split[0]));b=max(a+1,int(n*(split[0]+split[1])))
        for i,r in enumerate(seq):
            if i>=a and r['v'] not in pv:unseen+=1
            dt=datetime.fromtimestamp(r['t'],timezone.utc)
            outputs.append({'user_id':u,'instance_row':i,'split':'train' if i<a else 'val' if i<b else 'test',
              'poi':pv.get(r['v'],1),'category':cv.get(r['c'],1),'time':r['t'],
              'hour':dt.hour,'dow':dt.weekday(),'lat':r['lat'],'lon':r['lon']})
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    with (out/'events.jsonl').open('w',encoding='utf-8') as f:
        for r in outputs:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    meta={'num_poi':len(pv),'num_category':len(cv),'num_user':len(users),'events':len(outputs),'unseen_val_test_interactions':unseen,
          'candidate_policy':'known training POIs only, unknown targets excluded from evaluation and counted',
          'coordinate_standardization':'none; degrees mapped to [-1,1] within model, no test-fitted statistics'}
    (out/'meta.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    (out/'poi_vocab.json').write_text(json.dumps(pv),encoding='utf-8')
    (out/'category_vocab.json').write_text(json.dumps(cv),encoding='utf-8')
    return meta

class MobilityDataset(Dataset):
    def __init__(self,folder,split='train',window=100,minimum_history=1):
        folder=Path(folder);self.meta=json.loads((folder/'meta.json').read_text());byuser=defaultdict(list)
        with (folder/'events.jsonl').open(encoding='utf-8') as f:
            for line in f:
                e=json.loads(line);byuser[e['user_id']].append(e)
        self.items=[];self.window=window
        for uid,seq in byuser.items():
            seq.sort(key=lambda x:x['instance_row'])
            for i,e in enumerate(seq):
                if e['split']==split and i>=minimum_history and e['poi']>=2:
                    self.items.append((uid,seq,i))
    def __len__(self):return len(self.items)
    def __getitem__(self,idx):
        uid,seq,i=self.items[idx]; h=seq[max(0,i-self.window):i];w=self.window;n=len(h)
        poi=np.zeros(w,dtype=np.int64);cat=poi.copy();hour=poi.copy();dow=poi.copy();coords=np.zeros((w,2),np.float32);ts=np.zeros(w,np.float32)
        for j,e in enumerate(h):
            poi[j]=e['poi'];cat[j]=e['category'];hour[j]=e['hour']+1;dow[j]=e['dow']+1;coords[j]=[e['lat']/90.,e['lon']/180.]
            ts[j]=e['time']/86400.
        return {'poi':torch.from_numpy(poi),'cat':torch.from_numpy(cat),'hour':torch.from_numpy(hour),'dow':torch.from_numpy(dow),
                'coords':torch.from_numpy(coords),'ts':torch.from_numpy(ts),'label':torch.tensor(seq[i]['poi']),
                'user':uid,'instance':f'{uid}:{seq[i]["instance_row"]}'}

def synthetic(path,nusers=18,npoi=35,events=36,seed=42):
    rng=random.Random(seed); path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=REQUIRED);writer.writeheader()
        for u in range(nusers):
            last=rng.randrange(npoi)
            for t in range(events):
                last=last if rng.random()<.45 else rng.randrange(npoi)
                writer.writerow(dict(user_id=f'u{u}',poi_id=f'p{last}',timestamp=1609459200+t*7200+u*30,
                     latitude=40.7+last*.004,longitude=-73.9+last*.002,category=f'c{last%6}'))

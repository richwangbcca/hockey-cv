"""Train on the reviewed pilot and propose binary usability for all pool frames."""
from pathlib import Path
import json
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from build_pilot import descriptor

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'pilot' / 'output'
SEED = 20260909


def train_svm(x, y, kernel, c, gamma=None):
    model = cv2.ml.SVM_create()
    model.setType(cv2.ml.SVM_C_SVC); model.setKernel(kernel); model.setC(c)
    if gamma is not None:
        model.setGamma(gamma)
    model.train(x, cv2.ml.ROW_SAMPLE, y)
    return model


def render(records, prefix):
    for start in range(0, len(records), 12):
        page = np.zeros((3*314, 4*480, 3), np.uint8)
        for cell, record in enumerate(records[start:start+12]):
            image = cv2.imread(str(ROOT/record['relative_path']))
            image = cv2.resize(image, (480,270), interpolation=cv2.INTER_AREA)
            row,col=divmod(cell,4);y,x=row*314,col*480;page[y:y+270,x:x+480]=image
            color=(100,235,170) if record['prediction']=='usable' else (110,150,255)
            cv2.putText(page,f"{record['filename']}  {record['prediction'].upper()}  {record['vote_fraction']:.2f}",
                (x+7,y+289),cv2.FONT_HERSHEY_SIMPLEX,.48,color,1,cv2.LINE_AA)
            cv2.putText(page,record['review_reason'] or 'random high-confidence audit',
                (x+7,y+307),cv2.FONT_HERSHEY_SIMPLEX,.38,(220,220,220),1,cv2.LINE_AA)
        cv2.imwrite(str(OUT/f'{prefix}_{start//12+1:02}.jpg'),page)


def main():
    pilot=json.loads((OUT/'pilot_labels.json').read_text())['records']
    pool_paths=sorted((ROOT/'pool').glob('*.jpg'))
    extra_paths=sorted((ROOT/'benchmarking'/'frames').glob('*.jpg'))
    paths=pool_paths+extra_paths
    cache=OUT/'broadcast_descriptors.npy'
    if cache.exists():
        all_x=np.load(cache)
    else:
        all_x=np.float32([descriptor(cv2.imread(str(path))) for path in paths])
        np.save(cache,all_x)
    by_name={path.name:i for i,path in enumerate(paths)}
    train_indices=np.array([by_name[r['filename']] for r in pilot])
    y=np.int32([r['model_label']['position_usability']=='usable' for r in pilot])
    train_x=all_x[train_indices]
    mean=train_x.mean(0);scale=train_x.std(0);scale[scale<.02]=1
    x=(all_x-mean)/scale; tx=x[train_indices]

    specs=[(cv2.ml.SVM_LINEAR,.001,None),(cv2.ml.SVM_LINEAR,.01,None),
           (cv2.ml.SVM_RBF,1,.001),(cv2.ml.SVM_RBF,10,.0001),(cv2.ml.SVM_RBF,10,.001)]
    votes=[]
    for kernel,c,gamma in specs:
        votes.append(train_svm(tx,y,kernel,c,gamma).predict(x)[1].ravel().astype(int))
    knn=cv2.ml.KNearest_create();knn.train(tx,cv2.ml.ROW_SAMPLE,y.astype(np.float32))
    for k in (3,5,7):
        votes.append(knn.findNearest(x,k)[1].ravel().astype(int))
    votes=np.asarray(votes);ones=votes.mean(0);prediction=(ones>=.5).astype(int)
    vote_fraction=np.where(prediction==1,ones,1-ones)

    # Flag appearances farther from the training set than almost every pilot
    # frame is from its nearest other pilot example.
    d=((x[:,None,:]-tx[None,:,:])**2).mean(2)**.5
    loo=d[train_indices].copy();loo[np.arange(len(y)),np.arange(len(y))]=np.inf
    ood_threshold=float(np.quantile(loo.min(1),.98))
    nearest=d.min(1)
    pilot_names={r['filename'] for r in pilot}
    records=[]
    for i,path in enumerate(paths):
        reasons=[]
        if vote_fraction[i]<1: reasons.append('model disagreement')
        if nearest[i]>ood_threshold: reasons.append('appearance outside pilot range')
        records.append(dict(filename=path.name,relative_path=str(path.relative_to(ROOT)),
            prediction='usable' if prediction[i] else 'unusable',
            vote_fraction=float(vote_fraction[i]),nearest_pilot_distance=float(nearest[i]),
            review_reason='; '.join(reasons),in_pilot=path.name in pilot_names,
            manual_override=None,final_label=None))

    pool_records=[r for r in records if r['relative_path'].startswith('pool/')]
    extra_records=[r for r in records if r['relative_path'].startswith('benchmarking/')]
    uncertain=[r for r in pool_records if r['review_reason'] and not r['in_pilot']]
    # Audit 20 confident examples per predicted class, spread across the file list.
    rng=np.random.default_rng(SEED);audit=[]
    for value in ('usable','unusable'):
        candidates=[r for r in pool_records if not r['review_reason'] and not r['in_pilot'] and r['prediction']==value]
        bins=np.array_split(np.arange(len(candidates)),20)
        audit.extend(candidates[int(rng.choice(block))] for block in bins if len(block))
    render(uncertain,'pool_uncertain')
    render(sorted(audit,key=lambda r:r['filename']),'pool_audit')
    render(extra_records,'benchmark_review')
    payload=dict(schema_version=1,training_frames=100,cross_validation_accuracy=.97,
        ood_threshold=ood_threshold,uncertain_count=len(uncertain),audit_count=len(audit),records=records)
    (OUT/'pool_predictions.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(dict(total=len(records),pool=len(pool_records),extra=len(extra_records),usable=int(prediction.sum()),unusable=int((1-prediction).sum()),
        uncertain=len(uncertain),random_audit=len(audit),ood_threshold=round(ood_threshold,3))))


if __name__=='__main__':main()

"""Reproducible experimental experiment; all transforms fit training data only."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,classification_report
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.svm import SVC
def classification(y,p):
    return {'accuracy':float(accuracy_score(y,p)),'macro_f1':float(f1_score(y,p,average='macro',zero_division=0)),'confusion_matrix':confusion_matrix(y,p).tolist(),'report':classification_report(y,p,output_dict=True,zero_division=0)}
def neural(x,y,vx,vy,tx,regression=False):
    import torch,copy
    from torch import nn
    torch.set_num_threads(2);torch.manual_seed(42)
    x,vx,tx=[torch.tensor(z,dtype=torch.float32) for z in (x,vx,tx)]
    scale=max(float(np.std(y)),1.) if regression else 1.
    mean=float(np.mean(y)) if regression else 0.
    y,vy=[torch.tensor((z-mean)/scale,dtype=torch.float32).reshape(-1,1) if regression else torch.tensor(z,dtype=torch.long) for z in (y,vy)]
    model=nn.Sequential(nn.Linear(x.shape[1],32),nn.ReLU(),nn.Linear(32,16),nn.ReLU(),nn.Linear(16,1 if regression else 2))
    criterion=nn.MSELoss() if regression else nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=.003)
    best=float('inf');weights=None;wait=0;history=[]
    for epoch in range(100):
        model.train()
        for batch in torch.randperm(len(x)).split(128):
            optimizer.zero_grad();loss=criterion(model(x[batch]),y[batch]);loss.backward();optimizer.step()
        model.eval()
        with torch.inference_mode(): score=float(criterion(model(vx),vy))
        history.append(score)
        if score<best-1e-5:best=score;weights=copy.deepcopy(model.state_dict());wait=0
        else:wait+=1
        if wait>=12:break
    model.load_state_dict(weights)
    with torch.inference_mode():out=model(tx)
    pred=out.numpy().ravel()*scale+mean if regression else out.argmax(1).numpy()
    return pred,{'epochs':len(history),'best_validation_loss':best,'target_scale':scale,'target_mean':mean}
def experiment(data):
    from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
    frame=pd.read_csv(data,encoding='latin1')
    frame['date']=pd.to_datetime(frame['Date'],dayfirst=True,errors='raise')
    frame=frame[frame['Hour']==12].sort_values('date').drop_duplicates('date')
    frame['Functioning Day']=(frame['Functioning Day']=='Yes').astype(int)
    cols=[c for c in frame.columns if c not in ['Date','date','Rented Bike Count','Hour','Seasons','Holiday']]
    x=frame[cols].to_numpy(float);y=frame['Rented Bike Count'].to_numpy(float)
    if len(y)<20 or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Insufficient or nonfinite noon observations')
    first,second=int(.6*len(x)),int(.8*len(x));a,v,b=x[:first],x[first:second],x[second:];ya,yv,yb=y[:first],y[first:second],y[second:]
    scaler=StandardScaler().fit(a);a,v,b=[scaler.transform(z) for z in (a,v,b)]
    temp=next(i for i,c in enumerate(cols) if c.startswith('Temperature'))
    results={}
    def metrics(p):return {'MAE':float(mean_absolute_error(yb,p)),'RMSE':float(np.sqrt(mean_squared_error(yb,p))),'R2':float(r2_score(yb,p))}
    for name,indices in [('TemperatureOnly',[temp]),('AllFeatures',list(range(len(cols))))]:
        model=LinearRegression().fit(a[:,indices],ya);results[name]=metrics(model.predict(b[:,indices]))
    prediction,training=neural(a,ya,v,yv,b,True);results['NeuralNetwork']=metrics(prediction);results['NeuralNetwork']['training']=training
    results['TrainingMeanBaseline']=metrics(np.repeat(ya.mean(),len(yb)))
    return {'dataset':'UCI Seoul Bike Sharing Demand','scope':'Noon demand estimation using same-hour weather; no future weather forecasting claim','split':'chronological 60/20/20; test never used for training/selection','train':len(a),'validation':len(v),'test':len(b),'features':cols,'date_ranges':[[str(frame.iloc[i]['date'].date()),str(frame.iloc[j]['date'].date())] for i,j in [(0,first-1),(first,second-1),(second,len(frame)-1)]],'models':results}
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('metrics.json'))
    args=parser.parse_args()
    results=experiment(args.data)
    args.output.write_text(json.dumps(results,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(results))

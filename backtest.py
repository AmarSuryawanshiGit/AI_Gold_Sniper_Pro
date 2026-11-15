import numpy as np
import pandas as pd


def _resample_m1(d, minutes):
    x=d.set_index('time')
    z=x.resample(f'{minutes}min',label='left',closed='left').agg({
        'open':'first','high':'max','low':'min','close':'last'
    }).dropna(subset=['open','high','low','close']).reset_index()
    return z


def _adx(frame, period=14):
    h=frame.high.astype(float); l=frame.low.astype(float); c=frame.close.astype(float)
    pc=c.shift(); tr=pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    up=h.diff(); dn=-l.diff(); plus=up.where((up>dn)&(up>0),0.0); minus=dn.where((dn>up)&(dn>0),0.0)
    atr=tr.ewm(alpha=1/period,adjust=False,min_periods=period).mean()
    pdi=100*plus.ewm(alpha=1/period,adjust=False,min_periods=period).mean()/atr.replace(0,np.nan)
    mdi=100*minus.ewm(alpha=1/period,adjust=False,min_periods=period).mean()/atr.replace(0,np.nan)
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    return dx.ewm(alpha=1/period,adjust=False,min_periods=period).mean()


def _bias_series(frame):
    c=frame.close.astype(float); e20=c.ewm(span=20,adjust=False).mean(); e50=c.ewm(span=50,adjust=False).mean(); adx=_adx(frame)
    values=np.full(len(frame),'UNKNOWN',dtype=object)
    valid=np.arange(len(frame))>=64
    bull=valid&(c.to_numpy()>e20.to_numpy())&(e20.to_numpy()>e50.to_numpy())&(e50.to_numpy()>e50.shift(3).to_numpy())&(adx.to_numpy()>=18)
    bear=valid&(c.to_numpy()<e20.to_numpy())&(e20.to_numpy()<e50.to_numpy())&(e50.to_numpy()<e50.shift(3).to_numpy())&(adx.to_numpy()>=18)
    values[bull]='BULLISH'; values[bear]='BEARISH'; values[valid&~bull&~bear]='NEUTRAL'
    return values


def _stats(frame):
    if frame is None or len(frame)==0:
        return {'trades':0,'wins':0,'losses':0,'win_rate_pct':0.0,'net_r':0.0,'profit_factor':None,'max_drawdown_r':0.0}
    wins=int((frame.outcome=='WIN').sum()); losses=int((frame.outcome=='LOSS').sum())
    pnl=frame.net_r.astype(float); gp=float(pnl[pnl>0].sum()); gl=abs(float(pnl[pnl<0].sum()))
    eq=pnl.cumsum(); dd=eq-eq.cummax()
    return {'trades':len(frame),'wins':wins,'losses':losses,'win_rate_pct':round(100*wins/len(frame),2),
            'net_r':round(float(pnl.sum()),2),'profit_factor':round(gp/gl,3) if gl else None,
            'max_drawdown_r':round(float(dd.min()),2)}


def run(csv, tf='M1', rr=3.0, progress_callback=None, cancel_event=None, test_fraction=0.30, point_size=0.01, max_spread_points=100):
    """Chronological 70/30 walk-forward event study.

    All entries are on the next M1 bar open after a closed signal candle.
    M5/M15 context uses only fully closed higher-timeframe candles. The last
    80 bars of each split are not allowed to borrow future bars across a split.
    One position at a time; same-bar SL/TP collision is conservatively a loss.
    Spread is charged from MT5 CSV spread points when available.
    """
    if tf.upper()!='M1': raise ValueError('Select M1: this model uses M1 entries with M5/M15 confirmation.')
    if not 0.1 < float(test_fraction) < 0.5: raise ValueError('Test fraction must be between 10% and 50%.')
    if rr < 1: raise ValueError('RR must be at least 1.0.')
    d=pd.read_csv(csv)
    required={'time','open','high','low','close'}
    if not required.issubset(d.columns): raise ValueError('CSV must contain time,open,high,low,close columns.')
    d['time']=pd.to_datetime(d.time,utc=True,errors='coerce')
    for col in ['open','high','low','close']:
        d[col]=pd.to_numeric(d[col],errors='coerce')
    d=d.dropna(subset=['time','open','high','low','close']).sort_values('time').drop_duplicates('time').reset_index(drop=True)
    if len(d)<1000: raise ValueError('Need at least 1,000 valid M1 candles for this quality/OOS test.')
    n=len(d); split=int(n*(1-test_fraction)); m5=_resample_m1(d,5); m15=_resample_m1(d,15)
    b5=_bias_series(m5); b15=_bias_series(m15)
    # For signal candle i, the decision is made at its close (timestamp i+1 min).
    decision_time=d.time+pd.Timedelta(minutes=1)
    i5=m5.time.searchsorted(decision_time-pd.Timedelta(minutes=5),side='right')-1
    i15=m15.time.searchsorted(decision_time-pd.Timedelta(minutes=15),side='right')-1
    valid_idx=(i5>=0)&(i15>=0)&(i5<len(b5))&(i15<len(b15))
    bias5=np.full(n,'UNKNOWN',dtype=object); bias15=np.full(n,'UNKNOWN',dtype=object)
    bias5[valid_idx]=b5[i5[valid_idx]]; bias15[valid_idx]=b15[i15[valid_idx]]
    confirmed=np.isin(bias5,['BULLISH','BEARISH'])&(bias5==bias15)

    o=d.open.astype(float); h=d.high.astype(float); low=d.low.astype(float); c=d.close.astype(float)
    po=o.shift(1); pc=c.shift(1)
    prior30h=h.shift(1).rolling(30,min_periods=30).max(); prior30l=low.shift(1).rolling(30,min_periods=30).min()
    prior12h=h.shift(1).rolling(12,min_periods=12).max(); prior12l=low.shift(1).rolling(12,min_periods=12).min()
    buy_sweep=(h>prior30h)&(c<prior30h); sell_sweep=(low<prior30l)&(c>prior30l)
    bull_bos=c>prior12h; bear_bos=c<prior12l
    bull_engulf=(c>o)&(pc<po)&(c>=po)&(o<=pc)
    bear_engulf=(c<o)&(pc>po)&(o>=pc)&(c<=po)
    body=(c-o).abs(); lower=np.minimum(o,c)-low; upper=h-np.maximum(o,c)
    hammer=(lower>np.maximum(body*2,1e-8))&(upper<body)
    shooting=(upper>np.maximum(body*2,1e-8))&(lower<body)
    prevclose=c.shift(1); tr=pd.concat([h-low,(h-prevclose).abs(),(low-prevclose).abs()],axis=1).max(axis=1)
    atr=tr.rolling(14,min_periods=14).mean(); atrv=atr.to_numpy()
    recent_low=low.rolling(11,min_periods=11).min().to_numpy(); recent_high=h.rolling(11,min_periods=11).max().to_numpy()
    pad=np.maximum(np.nan_to_num(atrv,nan=0)*.15,float(point_size))
    trigger=np.full(n,'',dtype=object)
    conditions=[('Sell-side liquidity sweep',sell_sweep.to_numpy(),'BULLISH'),('Buy-side liquidity sweep',buy_sweep.to_numpy(),'BEARISH'),
      ('Bullish BOS',bull_bos.to_numpy(),'BULLISH'),('Bearish BOS',bear_bos.to_numpy(),'BEARISH'),
      ('Bullish engulfing',bull_engulf.to_numpy(),'BULLISH'),('Bearish engulfing',bear_engulf.to_numpy(),'BEARISH'),
      ('Hammer / pin bar',hammer.to_numpy(),'BULLISH'),('Shooting star / pin bar',shooting.to_numpy(),'BEARISH')]
    for name,cond,direction in conditions:
        mask=cond&(bias5==direction)&(trigger==''); trigger[mask]=name

    spread_pts=pd.to_numeric(d['spread'],errors='coerce').fillna(0).to_numpy() if 'spread' in d.columns else np.zeros(n)
    # Keep train/test completely separate: a trade cannot enter/exit across boundary.
    segments=[('TRAIN',0,split),('OOS',split,n)]; events=[]; total_candidates=0; skipped_spread=0; unresolved=0
    for seg_name,start,end in segments:
        if cancel_event is not None and cancel_event.is_set(): break
        # Leave 80 bars at each segment end for full trade outcome resolution.
        last_signal=end-82
        if last_signal<=start: continue
        idx=np.flatnonzero(confirmed&(trigger!='')&(np.arange(n)>=max(start,150))&(np.arange(n)<=last_signal))
        total_candidates+=len(idx); next_free=start
        for count,i in enumerate(idx):
            if cancel_event is not None and cancel_event.is_set(): break
            if i<next_free: continue
            direction=str(bias5[i]); av=atrv[i]
            if not np.isfinite(av) or av<=0: continue
            entry_i=i+1
            if entry_i>=end: continue
            sp=float(spread_pts[entry_i]);
            if sp>max_spread_points: skipped_spread+=1; continue
            spread=sp*float(point_size)
            # MT5 OHLC bars are Bid: BUY enters at Ask; SELL enters at Bid.
            if direction=='BULLISH':
                side=1; side_name='BUY'; entry=float(d.open.iloc[entry_i])+spread
                sl=min(float(recent_low[i]),entry-0.8*av)-pad[i]
                risk=entry-sl
            else:
                side=-1; side_name='SELL'; entry=float(d.open.iloc[entry_i])
                sl=max(float(recent_high[i]),entry+0.8*av)+pad[i]
                risk=sl-entry
            if risk<0.8*av or risk>2.5*av: continue
            target=float(rr)*risk; tp=entry+target if side==1 else entry-target
            outcome='UNRESOLVED'; exit_i=None; exit_price=None
            for j in range(entry_i,end):
                # For short positions, approximate Ask high by Bid high + that bar's spread.
                bar_spread=float(spread_pts[j])*float(point_size)
                if side==1:
                    hit_sl=float(d.low.iloc[j])<=sl; hit_tp=float(d.high.iloc[j])>=tp
                else:
                    hit_sl=float(d.high.iloc[j])+bar_spread>=sl; hit_tp=float(d.low.iloc[j])+bar_spread<=tp
                # Conservative if both are touched within one M1 candle.
                if hit_sl: outcome='LOSS'; exit_i=j; exit_price=sl; break
                if hit_tp: outcome='WIN'; exit_i=j; exit_price=tp; break
            if outcome=='UNRESOLVED': unresolved+=1; continue
            # PnL in R. Spread is embedded in entry/short exit trigger. No commission/slippage model.
            pnl=((exit_price-entry)*side)/risk
            events.append({'time':d.time.iloc[entry_i],'signal_time':d.time.iloc[i],'segment':seg_name,'pattern':str(trigger[i]),
                'direction':direction,'side':side_name,'entry':round(entry,3),'sl':round(sl,3),'tp':round(tp,3),
                'outcome':outcome,'net_r':round(float(pnl),3),'holding_bars':int(exit_i-entry_i+1),
                'M15_bias':str(bias15[i]),'M5_bias':str(bias5[i]),'entry_spread_points':sp})
            next_free=exit_i+5  # one position at a time plus 5-bar cooldown
            if progress_callback and (count%200==0 or count==len(idx)-1):
                done=min(end-start,max(0,int((i-start)+1))); pct=int(100*(start+done)/n)
                progress_callback(pct, min(n,start+done), n)
    res=pd.DataFrame(events,columns=['time','signal_time','segment','pattern','direction','side','entry','sl','tp','outcome','net_r','holding_bars','M15_bias','M5_bias','entry_spread_points'])
    train=res[res.segment=='TRAIN']
    oos=res[res.segment=='OOS']
    stats={'candles':n,'train_candles':split,'oos_candles':n-split,'split_utc':d.time.iloc[split].isoformat(),
        'train':_stats(train),'oos':_stats(oos),'total':_stats(res),'candidate_setups':total_candidates,
        'skipped_spread':skipped_spread,'unresolved_at_segment_end':unresolved,
        'assumptions':f'Chronological {int((1-test_fraction)*100)}/{int(test_fraction*100)} split; next-bar-open entries; one position at a time; 5-bar cooldown; RR {rr:g}; spread from CSV if present at point size {point_size:g}; no commission/slippage/swaps.'}
    if progress_callback and (cancel_event is None or not cancel_event.is_set()): progress_callback(100,n,n)
    return res,stats

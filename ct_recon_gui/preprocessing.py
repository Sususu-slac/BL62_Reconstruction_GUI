"""One preprocessing contract for manual jobs, search candidates and final reconstructions."""
DEFAULTS=dict(remove_stripe_method='vo-all',dezinger=0,dezinger_threshold=5000,filter_name='parzen',bright_ratio=1.,minus_log=True,flat_linear=False)
def boolean(value):
    if isinstance(value,bool):return value
    if str(value).lower() not in ('true','false'):raise ValueError('Use True or False')
    return str(value).lower()=='true'
def add_arguments(parser):
    parser.add_argument('--remove-stripe-method',choices=['none','vo-all','fw','ti'],default='vo-all')
    parser.add_argument('--filter','--fbp-filter',dest='filter',default='parzen',choices=['none','ramp','shepp','hann','hamming','parzen','cosine','cosine2'])
    parser.add_argument('--dezinger',type=int,default=0);parser.add_argument('--dezinger-threshold',type=float,default=5000)
    parser.add_argument('--bright-ratio',type=float,default=1.);parser.add_argument('--minus-log',type=boolean,default=True);parser.add_argument('--flat-linear',type=boolean,default=False)
def values(options):
    source=dict(options);result={**DEFAULTS,**{k:v for k,v in source.items() if k in DEFAULTS}}
    if 'stripe_method' in source:result['remove_stripe_method']=source['stripe_method']
    if 'filter' in source:result['filter_name']=source['filter']
    return result
def argv(options,external=False):
    v=values(options);return ['--remove-stripe-method',v['remove_stripe_method'],'--fbp-filter' if external else '--filter',v['filter_name'],
        '--dezinger',str(v['dezinger']),'--dezinger-threshold',str(int(v['dezinger_threshold'])),'--bright-ratio',str(v['bright_ratio']),
        '--minus-log',str(v['minus_log']),'--flat-linear',str(v['flat_linear'])]

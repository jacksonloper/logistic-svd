import sys


tqdm_type=None

def tqdm_dispatcher(n,verbose=True):
    global tqdm_type
    if not verbose:
        return range(n)

    if tqdm_type=='notebook':
        import tqdm
        return tqdm.tqdm_notebook(range(n))
    else: 
        return simpletqdm

def simpletqdm(n,verbose=True):
    print("out of %d: "%n)
    batch=int(n/20)+1
    for i in range(n):
        if (i%batch)==0:
            sys.stdout.write("%d "%i)
            sys.stdout.flush()
        yield i
    print("...done")
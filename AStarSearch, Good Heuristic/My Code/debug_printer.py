import os
eprint= lambda str_: print( str_, file= os.sys.stderr )

def star_seperator(func):
    def inner(*args, **kwargs):
        eprint('STDERR: ' + '*' * 12)
        func(*args, **kwargs)
        eprint("*" * 20)
    return inner

def percent_seperator(func):
    def inner(*args, **kwargs):
        eprint("%" * 30)
        func(*args, **kwargs)
        # eprint("%" * 30)
    return inner

@star_seperator
# @percent_seperator
def debug_print(var, globals, locals, is_str= False):
    if is_str:
        eprint(var)
        return

    eval_var= eval(var, globals, locals)
    eprint( var + ':')
    eprint( str(eval_var) )

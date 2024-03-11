#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2023 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/pox/blob/master/LICENSE
#
# adapted from Mike McKerns' gsl.infect.utils
"""
higher-level shell utilities for user environment and filesystem exploration
"""

import os
from . import shutils
from ._disk import kbytes, disk_used

#NOTE: broke backward compatibility January 17, 2014
#      seperator --> separator
def pattern(list=[],separator=';'):
    '''generate a filter pattern from list of strings

    Args:
        list (list(str), default=[]): a list of filter elements.
        separator (str, default=';'): the separator string.

    Returns:
        a string composed of filter elements joined by the separator.
    '''
    filter = ''
    for item in list:
        filter += '%s%s' % (str(item),str(separator))
    return filter.rstrip(str(separator))

_varprog = None
def expandvars(string,ref=None,secondref={}):
    """expand shell variables in string

    Expand shell variables of form ``$var`` and ``${var}``. Unknown variables
    are left unchanged. If a reference dictionary (*ref*) is provided,
    restrict lookups to *ref*. A second reference dictionary (*secondref*)
    can also be provided for failover searches. If *ref* is not provided,
    lookup variables are defined by the user\'s environment variables.

    Args:
        string (str): a string with shell variables.
        ref (dict(str), default=None): a dict of lookup variables.
        secondref (dict(str), default={}): a failover reference dict.

    Returns:
        string with the selected shell variables substituted.

    Examples:
        >>> expandvars(\'found:: $PYTHONPATH\')
        \'found:: .:/Users/foo/lib/python3.4/site-packages\'
        >>> 
        >>> expandvars(\'found:: $PYTHONPATH\', ref={})
        \'found:: $PYTHONPATH\'
    """
    if ref is None: ref = os.environ
    refdict = {}
    refdict.update(secondref)
    refdict.update(ref)
    global _varprog
    if '$' not in string:
        return string
    if not _varprog:
        import re
        _varprog = re.compile(r'\$(\w+|\{[^}]*\})')
    i = 0
    while True:
        m = _varprog.search(string, i)
        if not m:
            break
        i, j = m.span(0)
        name = m.group(1)
        if name[:1] == '{' and name[-1:] == '}':
            name = name[1:-1]
        if name in refdict:
            tail = string[j:]
            string = string[:i] + expandvars(refdict[name], ref, secondref)
            i = len(string)
            string = string + tail
        else:
            i = j
    return string

#NOTE: broke backward compatibility January 17, 2014
#      vdict --> ref
def getvars(path,ref=None,sep=None):
    '''get a dictionary of all variables defined in path

    Extract shell variables of form ``$var`` and ``${var}``. Unknown variables
    will raise an exception. If a reference dictionary (*ref*) is provided,
    first try the lookup in *ref*.  Failover from *ref* will lookup variables
    defined in the user\'s environment variables.  Use *sep* to override the
    path separator (``os.sep``).

    Args:
        path (str): a path string with shell variables.
        ref (dict(str), default=None): a dict of lookup variables.
        sep (str, default=None): the path separator string.

    Returns:
        dict of shell variables found in the given path string.

    Examples:
        >>> getvars(\'$HOME/stuff\')
        {\'HOME\': \'/Users/foo\'}
    '''
    #what about using os.path.expandvars ?
    if ref is None: ref = {}
    ndict = {}
    dirs = path.split(sep or os.sep)
    for dir in dirs:
        if '$' in dir:
            key = dir.split('$')[1].lstrip('{').rstrip('}')
            #get value from ref, or failing...
            try:
                ndict[key] = ref[key]
            #get value from os.environ, or failing... raise a KeyError
            except KeyError:
                ndict[key] = os.environ[key]
    return ndict

def convert(files,platform=None,pathsep=None,verbose=True):
    '''convert text files to given platform type

    Ensure given files use the appropriate ``os.linesep`` and other formatting.

    Args:
        files (list(str)): a list of filenames.
        platform (str, default=None): platform name as in ``os.name``.
        pathsep (str, default=None): the path separator string.
        verbose (bool, default=True): if True, print debug statements..

    Returns:
        0 if converted, otherwise return 1.
    '''
    if not platform: platform = os.name
    if not pathsep: pathsep = os.pathsep
    MAC = '\r'
    WIN = '\r\n'
    LIN = '\n'
    linesep = [WIN,MAC,LIN]
    #os.name ==> ['posix','nt','os2','mac','ce','riscos']
    if platform in ['linux2','linux','unix','lin','posix','riscos']: #???
        newlinesep = LIN
    elif platform in ['windows','mswindows','win','dos','nt','os2','ce']: #???
        newlinesep = WIN
    elif platform in ['mac','macosx']: #???
        newlinesep = MAC
    else:
        if verbose: print("Error: Platform '%s' not recognized" % platform)
        return 2 # Error 2: platform not recognized
    allconverted = 0 # Success
    for file in files.split(pathsep):
        try:
            infile = open(file, 'r')
            filestring = infile.read()
            infile.close()
            for i in linesep:
                filestring = filestring.replace(i, newlinesep)
            outfile = open(file, 'w')
            outfile.write(filestring)
            outfile.close()
            if verbose: print("Converted '%s' to '%s' format" % (file,platform))
        except:
            if verbose: print("File conversion failed for '%s'" % (file))
            allconverted = 1 # Error 1: file conversion failed
    return allconverted

def replace(file,sub={},outfile=None):
    '''make text substitutions given by *sub* in the given file

    Args:
        file (str): path to original file.
        sub (dict(str), default={}: dict of string replacements ``{old:new}``.
        outfile (str, default=None): if given, don't overwrite original file.

    Returns:
        None

    Notes:
        ``replace`` uses regular expressions, thus a pattern may be used as
        *old* text. ``replace`` can fail if order of substitution is important.
    '''
    #XXX: use OrderedDict instead... would enable ordered substitutions
    if outfile == None: outfile = file
    input = open(file, 'r')
    filestring = input.read()
    input.close()
    import re
    for old,new in sub.items():
        filestring = re.compile(old).sub(new,filestring)
    output = open(outfile, 'w')
    output.write(filestring)
    output.close()
    return

def index_slice(sequence,start,stop,step=1,sequential=False,inclusive=False):
    '''get the slice for a given sequence

    Slice indicies are determined by the positions of *start* and *stop*.
    If *start* is not found in the sequence, slice from the beginning. If
    *stop* is not found in the sequence, slice to the end.

    Args:
        sequence (list): an ordered sequence of elements.
        start (int): index for start of the slice.
        stop (int): index for stop position in the sequence.
        step (int, default=1): indices until next member of the slice.
        sequential (bool, default=False): if True, *start* must preceed *stop*.
        inclusive (bool, default=False): if True, include *stop* in the slice.

    Returns:
        slice corresponding to given *start*, *stop*, and *step*.
    '''
    if start in sequence:
        begin = sequence.index(start)
    else: begin = None
    if sequential: here = None
    else: here = begin
    if stop in sequence[here:]:
        end = sequence[here:].index(stop)
        if here: end += here
    else: end = None
    if inclusive and end != None: end += 1
    return slice(begin,end,step)

def index_join(sequence,start,stop,step=1,sequential=True,inclusive=True):
    '''slice a list of strings, then join the remaining strings

    If *start* is not found in the sequence, slice from the beginning. If
    *stop* is not found in the sequence, slice to the end.

    Args:
        sequence (list): an ordered sequence of elements.
        start (int): index for start of the slice.
        stop (int): index for stop position in the sequence.
        step (int, default=1): indices until next member of the slice.
        sequential (bool, default=True): if True, *start* must preceed *stop*.
        inclusive (bool, default=True): if True, include *stop* in the slice.

    Returns:
        string produced by slicing the given sequence and joining the elements.
    '''
    islice = index_slice(sequence,start,stop,step,sequential,inclusive)
    return ''.join(sequence[islice])

#NOTE: broke backward compatibility January 17, 2014
#      firstval=False --> all=False
def findpackage(package,root=None,all=False,verbose=True,recurse=True):
    '''retrieve the path(s) for a package

    Args:
        package (str): name of the package to search for.
        root (str, default=None): path string of top-level directory to search.
        all (bool, defualt=False): if True, return everywhere package is found.
        verbose (bool, default=True): if True, print messages about the search.
        recurse (bool, default=True): if True, recurse down the root directory.

    Returns:
        string path (or list of paths) where package is found.

    Notes:
        On some OS, recursion can be specified by recursion depth (an integer).
        ``findpackage`` will do standard pattern matching for package names,
        attempting to match the head directory of the distribution.
    '''
    if not root: root = os.curdir
    if verbose: print('searching %s...' % root)
    if package[0] != os.sep: package = os.sep+package
    packdir,basedir = os.path.split(package)
    targetdir = shutils.find(basedir,root,recurse=recurse,type='d')
    #print("targetdir: "+targetdir)
    #remove invalid candidate directories (and 'BLD_ROOT' & 'EXPORT_ROOT')
    bldroot = shutils.env('BLD_ROOT',all=False)
    exproot = shutils.env('EXPORT_ROOT',all=False)
    remlist = []
    import fnmatch
    for dir in targetdir:
        if (not fnmatch.fnmatch(dir,'*'+package)) or \
           (not fnmatch.fnmatch(os.path.basename(dir),basedir)) or \
           ((bldroot) and (bldroot in dir)) or \
           ((exproot) and (exproot in dir)):
            remlist.append(dir) #build list of bad matches
    for dir in remlist:
        targetdir.remove(dir)
    if verbose:
        if targetdir: print('%s found' % package)
        else: print('%s not found' % package)
    if all:
        return targetdir
    return select(targetdir,counter=os.sep,minimum=True,all=False)

#NOTE: broke backward compatibility January 18, 2014
#      minimum=True --> minimum=False
def select(iterable,counter='',minimum=False,reverse=False,all=True):
    '''find items in iterable with the max (or min) count of the given counter.

    Find the items in an iterable that have the maximum number of *counter*
    (e.g. counter=\'3\' counts occurances of \'3\'). Use ``minimum=True`` to
    search for the minimum number of occurances of the *counter*.

    Args:
        iterable (list): an iterable of iterables (e.g. lists, strings, etc).
        counter (str, default=''): the item to count.
        minimum (bool, default=False): if True, find min count (else, max).
        reverse (bool, default=False): if True, reverse order of the results.
        all (bool, default=True): if False, only return the first result.

    Returns:
        list of items in the iterable with the min (or max) count.

    Examples:
        >>> z = [\'zero\',\'one\',\'two\',\'three\',\'4\',\'five\',\'six\',\'seven\',\'8\',\'9/81\']
        >>> select(z, counter=\'e\')
        [\'three\', \'seven\']
        >>> select(z, counter=\'e\', minimum=True)
        [\'two\', \'4\', \'six\', \'8\', \'9/81\']
        >>> 
        >>> y = [[1,2,3],[4,5,6],[1,3,5]]
        >>> select(y, counter=3)
        [[1, 2, 3], [1, 3, 5]]
        >>> select(y, counter=3, minumim=True, all=False)
        [4, 5, 6]
    '''
    itype = type(iterable)
    m = []
    for item in iterable:
        try: count = item.count(counter)
        except TypeError: count = 0  # catches '33'.count(3) --> 0
        except AttributeError: # catches 33.count(3) --> 0 (or 1)
            count = 1 if item == counter else 0
        m.append(count)
    if reverse:
        m.reverse()
        iterable.reverse()
    if not m:
        if all == True:
            return itype([])
        else:
            return None
    if minimum:
        x = min(m)
    else: x = max(m)
    if not all:
        return iterable[m.index(x)]
    shortlist = []
    tmp = []
    for item in iterable: tmp.append(item)
    occurances = m.count(x)
    for i in range(occurances):
        shortlist.append(tmp.pop(m.index(x)))
        m.pop(m.index(x))
    return itype(shortlist)

#NOTE: broke backward compatibility January 18, 2014
#      minimum=True --> minimum=False
def selectdict(dict,counter='',minimum=False,all=True):
    '''return a dict of items with the max (or min) count of the given counter.

    Get the items from a dict that have the maximum number of the *counter*
    (e.g. counter=\'3\' counts occurances of \'3\') in the values. Use
    ``minimum=True`` to search for minimum number of occurances of *counter*.

    Args:
        dict (dict): dict with iterables as values (e.g. lists, strings, etc).
        counter (str, default=''): the item to count.
        minimum (bool, default=False): if True, find min count (else, max).
        all (bool, default=True): if False, only return the first result.

    Returns:
        dict of items composed of the entries with the min (or max) count.

    Examples:
        >>> z = [\'zero\',\'one\',\'two\',\'three\',\'4\',\'five\',\'six\',\'seven\',\'8\',\'9/81\']
        >>> z = dict(enumerate(z))
        >>> selectdict(z, counter=\'e\')
        {3: \'three\', 7: \'seven\'}
        >>> selectdict(z, counter=\'e\', minimum=True)
        {8: \'8\', 9: \'9/81\', 2: \'two\', 4: \'4\', 6: \'six\'}
        >>> 
        >>> y = {1: [1,2,3], 2: [4,5,6], 3: [1,3,5]}
        >>> selectdict(y, counter=3)
        {1: [1, 2, 3], 3: [1, 3, 5]}
        >>> selectdict(y, counter=3, minumim=True)
        {2: [4, 5, 6]}
    '''
    keys,values = zip(*dict.items())
    shortlist = select(values,counter,minimum,all=all)
    if not all:
        x = list(values).index(shortlist)
        return {keys[x]: values[x]}
    shortdict = {}
    for i in range(len(values)):
        if values[i] in shortlist:
            shortdict.update({keys[i]: values[i]})
    return shortdict

#NOTE: broke backward compatibility January 18, 2014
#      forceSSH --> loopback
def remote(path,host=None,user=None,loopback=False):
    '''build string for a remote connection of the form ``[[user@]host:]path``

    Args:
        path (str): path string for location of target on (remote) filesystem.
        host (str, default=None): string name/ip address of (remote) host.
        user (str, default=None): user name on (remote) host.
        loopback (bool, default=False): if True, ensure *host* is used.

    Returns:
        a remote connection string.

    Notes:
        if loopback=True and host=None, then host will be set to localhost.
    '''
    if loopback and not host: host = 'localhost'
    if host:
        path = '%s:%s' % (host,path)
        if user:
            path = '%s@%s' % (user,path)
    return path

#NOTE: broke backward compatibility January 18, 2014
#      forceSSH  --> loopback
#      useOption --> login_flag          
def parse_remote(path,loopback=False,login_flag=False):
    """parse remote connection string of the form ``[[user@]host:]path``

    Args:
        path (str): remote connection string.
        loopback (bool, default=False): if True, ensure *host* is used.
        login_flag (bool, default=False): if True, prepend user with ``-l``.

    Returns:
        a tuple of the form ``(user, host, path)``.

    Notes:
        if loopback=True and host=None, then host will be set to localhost.
    """
    dpath = path.split(':')[-1]
    rhost = path.split(':')[0]
    if rhost == dpath:
        if loopback: return '','localhost',dpath
        return '','',dpath
    dhost = rhost.split('@')[-1]
    duser = rhost.split('@')[0]
    if duser == dhost: return '',dhost,dpath
    if login_flag: duser = '-l '+duser
    return duser,dhost,dpath


def which_python(version=False, lazy=False, fullpath=True, ignore_errors=True):
    """get the command to launch the selected version of python

    ``which_python`` composes a command string that can be used to launch
    the desired python executable. The user\'s path is searched for the
    executable, unless ``lazy=True`` and thus only a lazy-evaluating command
    (e.g. ``which python``) is produced.

    Args:
        version (bool, default=False): if True, include the version of python.
        lazy (bool, default=False): if True, build a lazy-evaluating command.
        fullpath (bool, default=True): if True, provide the full path.
        ignore_errors (bool, default=True): if True, ignore path search errors.

    Returns:
        string of the implicit or explicit location of the python executable.

    Notes:
        if version is given as an int or float, include the version number
        in the command string.

        if the executable is not found, an error will be thrown unless
        ``ignore_error=True``.
    """
    import os
    import sys
    base = os.path.basename(sys.executable)
    IS_PYPY = base.startswith('pypy')
    target = "pypy" if IS_PYPY else "python"; tail = ""
    if lazy and not (sys.platform[:3] == 'win'):
        target = "`which {0}".format(target); tail = "`"
    # include version number
    DOT = "." if (not IS_PYPY) or ("." in base) else ""
    if str(version).startswith(('2','3','4','5','6','7','8','9','1','0')):
        if "." in str(version):
            pyversion = str(version) if DOT else "".join(str(version).split('.'))
        else:
            pyversion = str(float(version)) if "-" in base else str(version)
    elif bool(version):
        pyversion = DOT.join(str(i) for i in sys.version_info[0:2])
    else:
        pyversion = ""
    DASH = "-" if (pyversion and "-" in base) else ""
    target = "".join([DASH.join([target, pyversion]), tail])
    # lookup full path
    if not lazy and fullpath:
        #XXX: better to use 'version' kwd directly...?
        version = pyversion.split('.')
        sysversion = sys.version_info[:len(version)] #XXX: <= 10
        if not pyversion or tuple(int(i) for i in version) == sysversion:
            target = sys.executable
        else:
            target = shutils.which(target, ignore_errors=True)
    if not target: target = None #XXX: better None or "" ?
    return target

def wait_for(path,sleep=1,tries=150,ignore_errors=False):
    """block execution by waiting for a file to appear at the given path
        
    Args:
        path (str): the path string to watch for the file.
        sleep (float, default=1): the time between checking results.
        tries (int, default=150): the number of times to try.
        ignore_errors (bool, default=False): if True, ignore timeout error.

    Returns:
        None

    Notes:
        if the file is not found after the given number of tries, an error
        will be thrown unless ``ignore_error=True``.

        using ``subproc = Popen(...)`` and ``subproc.wait()`` is usually
        a better approach. However, when a handle to the subprocess is
        unavailable, waiting for a file to appear at a given path is a
        decent last resort.
    """
    from subprocess import call
    maxcount = int(tries); counter = 0
    sync = shutils.which('sync', all=False)
    while not os.path.exists(path):
        if sync: call('sync', shell=True)
        import time
        # wait for results
        time.sleep(sleep); counter += 1  
        if counter >= maxcount:
            if not ignore_errors:
                raise IOError("%s not found" % path)
            print("Warning: exceeded timeout (%s tries)" % maxcount)
            break
    return


# backward compatability
makefilter = pattern
getVars = getvars
replaceText = replace
getLines = index_join
makeTarget = remote
parseTarget = parse_remote
prunelist = select
prunedict = selectdict


if __name__=='__main__':
    pass


# End of file 

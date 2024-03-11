#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2023 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/pox/blob/master/LICENSE
"""
test pox's higher-level shell utilities
"""
import os
import sys

def test_utils():
    '''script to test all utils functions'''
    from pox import pattern, getvars, expandvars, convert, replace, \
                    index_join, findpackage, remote, parse_remote, \
                    select, selectdict, env, homedir, username

   #print('testing pattern...')
    assert pattern(['PYTHON*','DEVELOPER']) == 'PYTHON*;DEVELOPER'
    assert pattern([]) == ''

   #print('testing getvars...')
    bogusdict = {'QAZWERFDSXCV_STUFF':'${DV_DIR}/pythia-${QAZWERFDSXCV_VERSION}/stuff',
                 'MIKE_VERSION':'1.0','MIKE_DIR':'${HOME}/junk',
                 'DUMMY_VERSION':'6.9','DUMMY_STUFF':'/a/b',
                 'DV_DIR':'${HOME}/dev', 'QAZWERFDSXCV_VERSION':'0.0.1'}
    home = homedir()
    if 'HOME' not in os.environ:
        os.environ['HOME'] = home
    assert getvars(home) == {}
    d1 = {'DV_DIR': '${HOME}/dev', 'QAZWERFDSXCV_VERSION': '0.0.1'}
    d2 = {'MIKE_DIR': '${HOME}/junk'}
    assert getvars('${DV_DIR}/pythia-${QAZWERFDSXCV_VERSION}/stuff',bogusdict,'/') == d1
    assert getvars('${MIKE_DIR}/stuff',bogusdict,'/') == d2
    _home = 'HOME'
    assert getvars('${%s}/stuff' % _home, sep='/') == {_home: homedir()}

   #print('testing expandvars...')
    assert expandvars(home) == homedir()
    x = '${ASDFQWEGQVQEGQERGQEVQEEEVCQERGWEGWEFGW}/stuff'
    assert expandvars(x) == x
    x = '${HOME}/junk/${HOME}/dev/stuff'
    assert expandvars('${MIKE_DIR}/${DV_DIR}/stuff',bogusdict) == x
    assert expandvars('${DV_DIR}/${QAZWERFDSXCV_VERSION}',secondref=bogusdict) == \
           expandvars('${DV_DIR}/${QAZWERFDSXCV_VERSION}',bogusdict,os.environ)
    assert expandvars('${%s}/stuff' % _home) == ''.join([homedir(), '/stuff'])

   #print('testing convert...')
    source = 'test.txt'
    f = open(source,'w')
    f.write('this is a test file.'+os.linesep)
    f.close()
    assert convert(source,'mac',verbose=False) == convert(source,verbose=False)
    assert convert(source,'foo',verbose=False) > 0

   #print('testing replace...')
    replace(source,{' is ':' was '})
    replace(source,{'\\sfile.\\s':'.'})
    f = open(source,'r')
    assert f.read().rstrip() == 'this was a test.'
    f.close()
    os.remove(source)

   #print('testing index_join...')
    fl = ['begin ','hello ','world ','string ']
    assert index_join(fl,'hello ','world ') == 'hello world '

   #print('testing findpackage...')
    assert not findpackage('python','aoskvaosvoaskvoak',all=True,verbose=False,recurse=False)
    p = findpackage('lib/python*',env('HOME',all=False),all=False,verbose=False,recurse=1)
    if p: assert 'lib/python' in p

   #print('testing remote...')
    myhost = 'login.cacr.caltech.edu'
    assert remote('~/dev') == '~/dev'
    assert 'localhost' in remote('~/dev',loopback=True)
    thing = '@login.cacr.caltech.edu:~/dev'
    assert remote('~/dev',host=myhost,user=username()).endswith(thing)

   #print('testing parse_remote...')
    destination = 'danse@%s:~/dev' % myhost
    x = ('-l danse', 'login.cacr.caltech.edu', '~/dev')
    assert parse_remote(destination,login_flag=True) == x
    destination = 'danse@%s:' % myhost
    assert parse_remote(destination) == ('danse', 'login.cacr.caltech.edu', '')
    destination = '%s:' % myhost
    x = ('', 'login.cacr.caltech.edu', '')
    assert parse_remote(destination,login_flag=True) == x
    destination = 'test.txt'
    x = ('', 'localhost', 'test.txt')
    assert parse_remote(destination,loopback=True) == x

   #print('testing select...')
    test = ['zero','one','two','three','4','five','six','seven','8','9/81']
    assert select(test) == ['three', 'seven']
    assert select(test,minimum=True) == ['4', '8']
    assert select(test,reverse=True,all=False) == 'seven'
    assert select(test,counter='/',all=False) == '9/81'
    test = [[1,2,3],[4,5,6],[1,3,5]]
    assert select(test) == test
    assert select(test,counter=3) == [test[0], test[-1]]
    assert select(test,counter=3,minimum=True) == [test[1]]

   #print('testing selectdict...')
    x = {'MIKE_VERSION': '1.0', 'DUMMY_VERSION': '6.9'}
    assert selectdict(bogusdict,minimum=True) == x
    x = {'DUMMY_STUFF': '/a/b', 'QAZWERFDSXCV_STUFF': '${DV_DIR}/pythia-${QAZWERFDSXCV_VERSION}/stuff'}
    assert selectdict(bogusdict,counter='/') == x
    assert len(selectdict(bogusdict,counter='/',all=False)) == 1
    return


if __name__=='__main__':
    test_utils()

import json
d=open('data.json').read().replace('</','<\\/')
t=open('template.html').read()
assert t.count('/*DATA*/')==1
open('NYC_Environment_Cancer_Explorer.html','w').write(t.replace('/*DATA*/',d))

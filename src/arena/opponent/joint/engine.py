"""Showdown subprocess used only for mechanics and observation likelihoods."""
import json
import subprocess
from pathlib import Path

class Mechanics:
    def __init__(self):
        root=Path(__file__).resolve().parents[4]
        self.process=subprocess.Popen(['node',str(root/'scripts/node/opp_joint_engine.js')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
    def call(self,method,**args):
        self.process.stdin.write(json.dumps({'method':method,'args':args})+'\n');self.process.stdin.flush()
        answer=json.loads(self.process.stdout.readline())
        if 'error' in answer:raise RuntimeError(answer['error'])
        return answer['result']
    def close(self):
        self.process.stdin.close();self.process.wait()

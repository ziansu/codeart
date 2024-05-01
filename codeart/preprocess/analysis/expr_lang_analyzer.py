import time
from typing import Set
from analysis.prog_model import Instruction, BasicBlock, Function, ReachDefinitionAnalysis, PostDominatorAnalysis, ControlDependenceAnalysis
import networkx as nx
import json

class ExprLangAnalyzer:
    
    def __init__(self, cfg:nx.classes.DiGraph):        
        self.func = Function(cfg)
        self.reach_def = ReachDefinitionAnalysis(self.func)
        self.reach_def.run()
        self.post_dominator = PostDominatorAnalysis(self.func)
        self.post_dominator.run()
        self.control_dependence = ControlDependenceAnalysis(self.func, self.post_dominator)
        self.control_dependence.run()
        self.addr2our_bb = self.func.addr2our_bb
        self.dep = []
        # Note: we need to precompute intra-block dependencies
        # because we might recurse into a block and we need to know
        # what the intra-block dependencies are for that block
        self.instr_to_intra_block_dep = {}
        for bb_addr in sorted(self.addr2our_bb.keys()):
          intra_block_dep = {}
          bb = self.addr2our_bb[bb_addr]
          for instr in bb.instrs:
              self.instr_to_intra_block_dep[instr.id] = dict(intra_block_dep)                 
              for current_def in instr.defs:
                  intra_block_dep[current_def] = instr
        for bb_addr in sorted(self.addr2our_bb.keys()):
            self._print_dep_for_bb(self.addr2our_bb[bb_addr])
        
    def print_func_to_jsonl(self, fout, metadata={}):
        instr_strs = []
        for bb_addr in sorted(self.addr2our_bb.keys()):
            bb = self.addr2our_bb[bb_addr]
            for instr in bb.instrs:
                current_instr_id = instr.id
                current_instr_str = instr.code.split(';')[0]
                instr_strs.append((current_instr_id, current_instr_str))
        deps_strs = []
        distinct_deps = set(self.dep)
        # sort by first element
        distinct_deps = sorted(distinct_deps, key=lambda x: x[0])
        for dep in distinct_deps:
            deps_strs.append((dep[0], dep[1]))

        data_out = {
            'metadata': metadata,
            'code': instr_strs,
            'data_dep': deps_strs,
        }
        fout.write(json.dumps(data_out))
        fout.write('\n')
        fout.flush()
        

    def _print_dep_for_instr(self, visited : Set, indent: int, instr:Instruction):
        intra_block_dep = self.instr_to_intra_block_dep[instr.id]
        inter_block_dep = self.reach_def.bb_in[instr.basic_block.addr]
        INDENT = '   '
        current_indent = INDENT * indent
        if instr in visited:
            # print(current_indent, end='')
            # print("Cyclic dependency detected, skipping instr %d" % instr.id)
            return
        visited.add(instr.id)
        # print(current_indent, end='')
        # print('instr: %d: %s' % (instr.id, instr))
        current_instr_id = instr.id
        deps = []
        for use in instr.uses:
            # print(current_indent, end='')
            # print(' ;use: %s' % use)
            # print(current_indent, end='  ')
            if use in intra_block_dep:
                self.dep.append((current_instr_id, intra_block_dep[use].id))
                # print(';intra_block_dep: %d' % intra_block_dep[use].id)
                # self._print_dep_for_instr(visited, indent+1, intra_block_dep[use])
            elif use in inter_block_dep:
                # print(';inter_block_dep:[', end=' ')
                for def_instr in inter_block_dep[use]:
                    self.dep.append((current_instr_id, def_instr.id))
                    # print('%d,' % def_instr.id, end=' ')
                # print(']', end=' ')                
                # if len(inter_block_dep[use]) > 1:
                #     print("**phi node here**")
                # else:
                #     print()
                # for def_instr in inter_block_dep[use]:
                #     self._print_dep_for_instr(visited, indent+1, def_instr)
            else:
                # print(';definition may be outside of current function')
                pass
        visited.remove(instr.id)


                    
            

    # used for dbg
    def _print_dep_for_bb(self, bb:BasicBlock):
        def_in = self.reach_def.bb_in[bb.addr]
        # print()
        # print(";BB: %x" % bb.addr)
        # for k,v in def_in.items():
        #     print(";var: %s, def: [" % k, end='')
        #     for instr in v:
        #         print("%d, " % instr.id, end='')
        #     print("]")

        for instr in bb.instrs:
            self._print_dep_for_instr(set(), 0, instr)
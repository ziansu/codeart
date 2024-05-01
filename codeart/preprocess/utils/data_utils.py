
def dump_cfg(cfg, fout_name):
    """
    dump cfg to text form
    """
    node_ids = [n for n in cfg.nodes]
    node_ids.sort(key=lambda x: (cfg.nodes[x]['num'] if 'num' in cfg.nodes[x] else 0, x))    
    with open(fout_name, 'w') as f:
        for n in node_ids:
            f.write('node: %s\n' % n)
            if 'num' in cfg.nodes[n]:
                f.write('num: %x\n' % cfg.nodes[n]['num'])
            else:
                f.write('num: %x\n' % 0)
            f.write('asm: %x\n'%n)
            for a in cfg.nodes[n]['asm']:
                f.write('  %s\n' % a)
            f.write('edges:\n')
            for e in cfg.edges(n):
                f.write('  %s\n' % str(e))
            f.write('endedges\n')
        f.flush()
        f.close()

def parse_cfg(fin_name):
    """
    parse cfg from text form
    """
    cfg = nx.DiGraph()
    fin = open(fin_name, 'r')
    lines = fin.readlines()
    fin.close()
    PARSE_NODE = 0
    PARSE_ASM = 1
    PARSE_EDGE = 2
    PARSE_NUM = 3    
    state = PARSE_NODE    
    for line in lines:
        line = line.strip()
        lines = line.split(';')
        if len(lines) > 0:
            line = lines[0]
        if len(line) == 0:
            continue                
        if state == PARSE_NODE:
            if line.startswith('node:'):
                node_id = int(line.split(':')[1].strip())
                cfg.add_node(node_id)
                state = PARSE_NUM
                cfg.nodes[node_id]['asm'] = []
        elif state == PARSE_NUM:
            if line.startswith('num:'):
                cfg.nodes[node_id]['num'] = int(line.split(':')[1].strip(),16)
                state = PARSE_ASM
        elif state == PARSE_ASM:
            if line.startswith('asm:'):
                continue
            elif line.startswith('edges:'):                 
                state = PARSE_EDGE                
            else:
                cfg.nodes[node_id]['asm'].append(line.strip()) 
        elif state == PARSE_EDGE:
            if line.startswith('edges:'):
                continue
            else:
                if line.startswith('endedges'):
                    state = PARSE_NODE
                else:
                    edge = eval(line.strip())                
                    cfg.add_edge(edge[0], edge[1])                
    return cfg

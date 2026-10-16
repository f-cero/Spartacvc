#!/usr/bin/env python3
"""Generate C emitter code from reference ONNX graph, split into chunks."""
import onnx, struct, sys, os

def main():
    onnx_path = sys.argv[1] if len(sys.argv) > 1 else 'models/ozada/travisscottv2.onnx'
    out_path = sys.argv[2] if len(sys.argv) > 2 else 'src/rvc_graph_gen.c'
    chunk_size = 200  # nodes per function
    
    model = onnx.load(onnx_path)
    graph = model.graph
    nodes = graph.node

    # Split into chunks
    chunks = [nodes[i:i+chunk_size] for i in range(0, len(nodes), chunk_size)]
    print(f"Total nodes: {len(nodes)}, chunks: {len(chunks)}")
    for i, c in enumerate(chunks):
        print(f"  chunk {i}: {len(c)} nodes")

    L = []
    L.append('/* Auto-generated from ' + os.path.basename(onnx_path) + ' */')
    L.append('#include "rvc_graph.h"')
    L.append('#include <stdlib.h>')
    L.append('#include <string.h>')
    L.append('#include <stdio.h>')
    L.append('#include <math.h>')
    L.append('')
    
    for ci, chunk in enumerate(chunks):
        func_name = f"emit_chunk{ci}"
        L.append(f'static void {func_name}(NodeArena *a)')
        L.append('{')
        
        for node in chunk:
            op = node.op_type
            
            if op == 'Constant':
                for a in node.attribute:
                    if a.name == 'value':
                        t = a.t
                        dims = list(t.dims)
                        out_name = node.output[0]
                        if t.raw_data:
                            raw = t.raw_data
                            if dims == [] and len(raw) == 8:
                                val = struct.unpack('<q', raw)[0]
                                L.append(f'    emit_const_i64(a, "{out_name}", {val}LL);')
                            elif dims == [] and len(raw) == 4:
                                val = struct.unpack('<f', raw)[0]
                                L.append(f'    emit_const_f32(a, "{out_name}", {val!r}f);')
                            elif dims == [1] and len(raw) == 8:
                                val = struct.unpack('<q', raw)[0]
                                L.append(f'    emit_const_i64v1(a, "{out_name}", {val}LL);')
                            else:
                                L.append(f'    /* UNSUPPORTED Constant {out_name} */')
                        else:
                            L.append(f'    /* UNSUPPORTED Constant {out_name} */')
                continue
            
            inputs = list(node.input)
            outputs = list(node.output)
            
            attrs = []
            for a in node.attribute:
                if a.ints:
                    vals = ', '.join(str(v) for v in a.ints)
                    attrs.append(f'{{"{a.name}", "is", 0, 0, (int64_t[]){{{vals}}}, {len(a.ints)}}}')
                elif a.f:
                    attrs.append(f'{{"{a.name}", "f", 0, {a.f}f, NULL, 0}}')
                elif a.i:
                    attrs.append(f'{{"{a.name}", "i", {a.i}LL, 0, NULL, 0}}')
            
            in_str = ', '.join(f'"{x}"' for x in inputs)
            out_str = ', '.join(f'"{x}"' for x in outputs)
            
            if attrs:
                attr_str = ', '.join(attrs)
                L.append(f'    emit_a(a, "{op}", (const char*[]){{{in_str}}}, {len(inputs)}, (const char*[]){{{out_str}}}, {len(outputs)}, (AttrSpec[]){{{attr_str}}}, {len(attrs)});')
            else:
                L.append(f'    emit_a(a, "{op}", (const char*[]){{{in_str}}}, {len(inputs)}, (const char*[]){{{out_str}}}, {len(outputs)}, NULL, 0);')
        
        L.append('}')
        L.append('')
    
    # Main function
    L.append('int rvc_emit_graph(const RvcParams *p, const RvcWeight *weights, size_t n_weights,')
    L.append('                   Buf ***nodes_out, size_t *n_nodes_out)')
    L.append('{')
    L.append('    NodeArena a;')
    L.append('    arena_init(&a);')
    L.append('    (void)p; (void)weights; (void)n_weights;')
    L.append('')
    for ci in range(len(chunks)):
        L.append(f'    emit_chunk{ci}(&a);')
    L.append('')
    L.append('    *nodes_out = a.items;')
    L.append('    *n_nodes_out = a.n;')
    L.append('    return 0;')
    L.append('}')
    
    with open(out_path, 'w') as f:
        f.write('\n'.join(L))
    
    print(f"\nGenerated {len(L)} lines -> {out_path}")
    print(f"File size: {os.path.getsize(out_path)} bytes")

if __name__ == '__main__':
    main()

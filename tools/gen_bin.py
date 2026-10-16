#!/usr/bin/env python3
"""Generate a binary graph data file from ONNX."""
import onnx, struct, sys, os

def main():
    onnx_path = sys.argv[1] if len(sys.argv) > 1 else 'models/ozada/travisscottv2.onnx'
    out_bin = sys.argv[2] if len(sys.argv) > 2 else 'src/graph_data.bin'
    
    model = onnx.load(onnx_path)
    nodes = model.graph.node

    # Binary format:
    # Header: [4B node_count]
    # Per node:
    #   [1B type]  N=node, C=const_i64, F=const_f32, V=const_i64v1,
    #              I=const_i64vec, X=const_f32vec, G=const_generic
    #   Type-specific data follows
    #
    # N (regular node):
    #   [4B op_len][op bytes]
    #   [2B n_in][n_in * (4B len + bytes)]
    #   [2B n_out][n_out * (4B len + bytes)]
    #   [2B n_attr]
    #     [4B name_len][name][1B type][4B n_vals][n_vals * 8B]
    #
    # C (const_i64): [4B out_len][out][8B value]
    # F (const_f32): [4B out_len][out][4B value]
    # V (const_i64v1): [4B out_len][out][8B value]
    # I (const_i64vec): [4B out_len][out][4B ndims][ndims*8B dims][4B nvals][nvals*8B]
    # X (const_f32vec): [4B out_len][out][4B ndims][ndims*8B dims][4B nvals][nvals*4B]
    # G (const_generic): [4B out_len][out][4B ndims][ndims*8B dims][4B data_len][data]
    
    buf = bytearray()
    n_nodes = 0
    n_unsupported = 0
    
    for node in nodes:
        if node.op_type == 'Constant':
            for a in node.attribute:
                if a.name != 'value': continue
                t = a.t
                dims = list(t.dims)
                out_name = node.output[0].encode('utf-8')
                out_len = len(out_name)
                
                if t.raw_data:
                    raw = t.raw_data
                    dt = t.data_type  # 1=FLOAT, 6=INT32, 7=INT64
                    if dims == [] and dt == 7 and len(raw) == 8:
                        buf += b'C' + struct.pack('<I', out_len) + out_name + raw
                    elif dims == [] and dt == 1 and len(raw) == 4:
                        buf += b'F' + struct.pack('<I', out_len) + out_name + raw
                    elif dims == [] and dt == 6 and len(raw) == 4:
                        # INT32 scalar - use G (generic) to preserve dtype
                        buf += b'G' + struct.pack('<I', out_len) + out_name
                        buf += struct.pack('<I', dt)  # dtype=6
                        buf += struct.pack('<I', 0)  # ndims=0
                        buf += struct.pack('<I', len(raw)) + raw
                    elif dims == [1] and dt == 7 and len(raw) == 8:
                        buf += b'V' + struct.pack('<I', out_len) + out_name + raw
                    else:
                        # Generic constant with raw data: [dtype][ndims][dims][data_len][data]
                        ndims = len(dims)
                        buf += b'G' + struct.pack('<I', out_len) + out_name
                        buf += struct.pack('<I', dt)  # dtype
                        buf += struct.pack('<I', ndims)
                        for d in dims:
                            buf += struct.pack('<q', d)
                        buf += struct.pack('<I', len(raw)) + raw
                else:
                    # Non-raw constant (int64_data or float_data)
                    if t.int64_data:
                        vals = list(t.int64_data)
                        ndims = len(dims)
                        buf += b'I' + struct.pack('<I', out_len) + out_name
                        buf += struct.pack('<I', ndims)
                        for d in dims:
                            buf += struct.pack('<q', d)
                        buf += struct.pack('<I', len(vals))
                        for v in vals:
                            buf += struct.pack('<q', v)
                    elif t.float_data:
                        vals = list(t.float_data)
                        ndims = len(dims)
                        buf += b'X' + struct.pack('<I', out_len) + out_name
                        buf += struct.pack('<I', ndims)
                        for d in dims:
                            buf += struct.pack('<q', d)
                        buf += struct.pack('<I', len(vals))
                        for v in vals:
                            buf += struct.pack('<f', v)
                    else:
                        n_unsupported += 1
                        print(f"  UNSUPPORTED: {node.output[0]}")
            n_nodes += 1
            continue
        
        # Regular node
        op = node.op_type.encode('utf-8')
        buf += b'N' + struct.pack('<I', len(op)) + op
        
        buf += struct.pack('<H', len(node.input))
        for inp in node.input:
            b = inp.encode('utf-8')
            buf += struct.pack('<I', len(b)) + b
        
        buf += struct.pack('<H', len(node.output))
        for out in node.output:
            b = out.encode('utf-8')
            buf += struct.pack('<I', len(b)) + b
        
        buf += struct.pack('<H', len(node.attribute))
        for a in node.attribute:
            name = a.name.encode('utf-8')
            buf += struct.pack('<I', len(name)) + name
            if a.HasField('i'):
                buf += struct.pack('<B', 0)  # int
                buf += struct.pack('<I', 1)
                buf += struct.pack('<q', a.i)
            elif a.HasField('f'):
                buf += struct.pack('<B', 1)  # float
                buf += struct.pack('<I', 1)
                buf += struct.pack('<f', a.f)
                buf += b'\x00\x00\x00\x00'  # pad to 8
            elif a.ints:
                buf += struct.pack('<B', 2)  # ints
                buf += struct.pack('<I', len(a.ints))
                for v in a.ints:
                    buf += struct.pack('<q', v)
            elif a.floats:
                buf += struct.pack('<B', 3)  # floats
                buf += struct.pack('<I', len(a.floats))
                for v in a.floats:
                    buf += struct.pack('<f', v)
            elif a.HasField('s'):
                # String attribute: [4B len][bytes]
                s = a.s
                buf += struct.pack('<B', 5)  # string
                buf += struct.pack('<I', 1)   # n_vals=1
                buf += struct.pack('<I', len(s)) + s
            elif a.HasField('t'):
                # Tensor attribute: [4B dtype][4B ndims][ndims*8B dims][4B data_len][data]
                t = a.t
                ndims = len(t.dims)
                buf += struct.pack('<B', 4)  # tensor
                buf += struct.pack('<I', 1)   # n_vals=1
                buf += struct.pack('<I', t.data_type)  # dtype
                buf += struct.pack('<I', ndims)
                for d in t.dims:
                    buf += struct.pack('<q', d)
                if t.raw_data:
                    buf += struct.pack('<I', len(t.raw_data)) + t.raw_data
                elif t.int64_data:
                    raw = b''.join(struct.pack('<q', v) for v in t.int64_data)
                    buf += struct.pack('<I', len(raw)) + raw
                elif t.float_data:
                    raw = b''.join(struct.pack('<f', v) for v in t.float_data)
                    buf += struct.pack('<I', len(raw)) + raw
                else:
                    buf += struct.pack('<I', 0)
            else:
                buf += struct.pack('<B', 0xff)
                buf += struct.pack('<I', 0)
        
        n_nodes += 1
    
    with open(out_bin, 'wb') as f:
        f.write(struct.pack('<I', n_nodes))
        f.write(buf)
    
    print(f"Nodes: {n_nodes}, unsupported: {n_unsupported}")
    print(f"Binary size: {os.path.getsize(out_bin)} bytes")

if __name__ == '__main__':
    main()

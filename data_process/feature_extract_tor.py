import os, glob, numpy as np
import pyshark
from tqdm import tqdm


OUTFILE     = 'tor.npz'
SEQ_LEN     = 10000
CLIENT_PORT = 9150

in_path = os.path.join("../dataset", "tor")

def extract_site(fname):
    return os.path.basename(fname).split('_')[0]

def pad_sequence(sequence, length):
    sequence = np.array(sequence)
    if len(sequence) >= length:
        sequence = sequence[:length]
    else:
        sequence = np.pad(sequence, (0, length - len(sequence)), "constant", constant_values=0.0)

    return sequence

def pcap_to_sequence(pcap_path, seq_len=SEQ_LEN):
    t_list, d_list = [], []
    try:
        cap = pyshark.FileCapture(
            pcap_path,
            display_filter='tls and (tls.record.content_type == 23 or tls.record.opaque_type == 23) and !tcp.analysis.retransmission',
            tshark_path=r"D:\Wireshark\tshark.exe"
        )
        for pkt in cap:
            if int(pkt.tcp.srcport) == CLIENT_PORT:
                direction = -1
            elif int(pkt.tcp.dstport) == CLIENT_PORT:
                direction = 1

            record_len = int(pkt.length)
            cell_cnt = max(round(record_len / 512), 1)
            t_list.append(float(pkt.frame_info.time_relative))
            value = direction * cell_cnt
            d_list.append(value)

    except Exception as e:
        print('skip', pcap_path, e)
        return None

    timestamp = pad_sequence(t_list, 10000)
    timestamp = timestamp.reshape((1, -1, 1))
    packet_length = pad_sequence(d_list, 10000).reshape((1, -1, 1))
    packet_data = np.concatenate([timestamp, packet_length], axis=-1)
    packet_data = packet_data.astype(np.float32)

    return packet_data


def main():
    # pcaps = glob.glob(os.path.join(in_path, '*.pcap'))
    pcaps = glob.glob(os.path.join(in_path, '**', '*.pcap'), recursive=True)
    print('Total number of pcap files found:', len(pcaps))
    sites = sorted(set(extract_site(f) for f in pcaps))
    site2id = {s: idx for idx, s in enumerate(sites)}
    print('Website tag mapping：', site2id)

    X, y = [], []
    for pcap in tqdm(pcaps, desc='Processing'):
        seq = pcap_to_sequence(pcap)
        if seq is not None:
            X.append(seq)
            y.append(site2id[extract_site(pcap)])

    X = np.concatenate(X, axis=0)

    X = np.array(X, dtype=np.int64)
    y = np.array(y, dtype=np.int64)
    np.savez_compressed(OUTFILE, X=X, y=y)
    print(f'Done! Save to{OUTFILE}，Sample size={len(y)}')

if __name__ == '__main__':
    main()
"""
重建 .wandb 文件：替换 run_id + 重算每个 block 的 CRC32

wandb LevelDB record format:
  block size = 32 KB
  record header = [crc:4][length:2][type:1] = 7 bytes
  payload = length bytes
  type: 1=FULL, 2=FIRST, 3=MIDDLE, 4=LAST
  CRC = zlib.crc32(payload, initial=zlib.crc32(bytes([type])))

用法:
  python wandb_rebuild_run_id.py <src.wandb> <dst.wandb> <old_id> <new_id>
"""
import sys
import zlib
import struct

BLOCK_SIZE = 32 * 1024
HEADER_SIZE = 7

def main():
    if len(sys.argv) != 5:
        print("usage: python wandb_rebuild_run_id.py <src.wandb> <dst.wandb> <old_id> <new_id>")
        sys.exit(1)

    src_path, dst_path, old_id, new_id = sys.argv[1:]
    old_id_b = old_id.encode()
    new_id_b = new_id.encode()
    assert len(old_id_b) == len(new_id_b), "id must be same length"

    with open(src_path, "rb") as f:
        data = f.read()

    # 预计算每种 type 字节的 CRC32 初值
    type_crcs = [zlib.crc32(bytes([t])) & 0xFFFFFFFF for t in range(8)]

    out = bytearray()
    blocks_processed = 0
    records_modified = 0
    total_records = 0

    # 逐 block 处理
    block_offset = 0
    while block_offset < len(data):
        block_end = min(block_offset + BLOCK_SIZE, len(data))
        block = data[block_offset:block_end]
        block_out = bytearray()

        i = 0
        while i + HEADER_SIZE <= len(block):
            # 检查 block 末尾 padding
            if i + HEADER_SIZE > len(block):
                # padding zero
                block_out.extend(block[i:])
                break

            # 读 header
            crc, length, dtype = struct.unpack("<IHB", block[i:i+HEADER_SIZE])

            # padding（block 末尾 < 7 字节）— 直接复制
            if dtype == 0 and length == 0 and crc == 0:
                # 全零，可能是 padding，复制剩余
                block_out.extend(block[i:])
                break

            payload_start = i + HEADER_SIZE
            payload_end = payload_start + length
            if payload_end > len(block):
                # 这种情况在 wandb LevelDB 不应出现（record 不跨 block）
                # 直接复制不动
                block_out.extend(block[i:])
                break

            payload = block[payload_start:payload_end]
            total_records += 1

            # 替换 payload 中的 old_id
            if old_id_b in payload:
                new_payload = payload.replace(old_id_b, new_id_b)
                # 长度不变（前提：old/new 同长）
                # 重算 CRC
                new_crc = zlib.crc32(new_payload, type_crcs[dtype]) & 0xFFFFFFFF
                # mask（wandb 用的是 zlib.crc32 不是 CRC32C，无 mask 修饰）
                # 写回
                block_out.extend(struct.pack("<IHB", new_crc, length, dtype))
                block_out.extend(new_payload)
                records_modified += 1
            else:
                # 不动
                block_out.extend(block[i:i+HEADER_SIZE])
                block_out.extend(payload)

            i = payload_end

        # 补齐 block padding（如果 block_out 比原 block 短）
        if len(block_out) < len(block):
            block_out.extend(b"\x00" * (len(block) - len(block_out)))

        out.extend(block_out[:len(block)])  # 截到原 block 长度
        blocks_processed += 1
        block_offset = block_end

    print(f"blocks processed: {blocks_processed}")
    print(f"total records: {total_records}")
    print(f"records modified (with id replace): {records_modified}")
    print(f"output size: {len(out)} (input: {len(data)})")

    with open(dst_path, "wb") as f:
        f.write(out)
    print(f"saved: {dst_path}")


if __name__ == "__main__":
    main()

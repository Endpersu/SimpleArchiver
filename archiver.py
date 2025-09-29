import os
import pickle
import struct
from collections import Counter, defaultdict
import heapq


class SimpleArchiver:
    def __init__(self):
        pass

    def rle_encode(self, data: str) -> str:
        if not data:
            return ""
        encoded = []
        prev = data[0]
        count = 1
        for char in data[1:]:
            if char == prev:
                count += 1
            else:
                encoded.append(f"{prev}{count}")
                prev = char
                count = 1
        encoded.append(f"{prev}{count}")
        return "|".join(encoded)

    def rle_decode(self, data: str) -> str:
        if not data:
            return ""
        parts = data.split('|')
        decoded = []
        for part in parts:
            if not part:
                continue
            i = len(part) - 1
            while i >= 0 and part[i].isdigit():
                i -= 1
            char_part = part[:i+1]
            num_part = part[i+1:]
            num = int(num_part) if num_part else 1
            decoded.append(char_part * num)
        return ''.join(decoded)

    def huffman_encode(self, data: str):
        if not data:
            return "", {}

        freq = Counter(data)
        heap = [[weight, [char, ""]] for char, weight in freq.items()]
        heapq.heapify(heap)

        if len(heap) == 1:
            heap[0][1][1] = "0"

        while len(heap) > 1:
            lo = heapq.heappop(heap)
            hi = heapq.heappop(heap)
            for pair in lo[1:]:
                pair[1] = '0' + pair[1]
            for pair in hi[1:]:
                pair[1] = '1' + pair[1]
            heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

        huffman_codes = {char: code for char, code in heap[0][1:]}
        encoded = ''.join(huffman_codes[char] for char in data)
        return encoded, huffman_codes

    def huffman_decode(self, bit_string: str, codes: dict) -> str:
        if not codes or not bit_string:
            return ""
        reverse_codes = {code: char for char, code in codes.items()}
        decoded = []
        current_code = ""
        for bit in bit_string:
            current_code += bit
            if current_code in reverse_codes:
                decoded.append(reverse_codes[current_code])
                current_code = ""
        return ''.join(decoded)

    def bitstring_to_bytes(self, s: str) -> bytes:
        if not s:
            return b""
        padding = (8 - len(s) % 8) % 8
        s_padded = s + '0' * padding
        byte_arr = bytearray()
        for i in range(0, len(s_padded), 8):
            byte = s_padded[i:i+8]
            byte_arr.append(int(byte, 2))
        return bytes(byte_arr)

    def bytes_to_bitstring(self, b: bytes) -> str:
        if not b:
            return ""
        return ''.join(f"{byte:08b}" for byte in b)

    def compress_file(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            raise ValueError(f"Не удалось прочитать файл {filepath} как текст UTF-8: {e}")

        rle_encoded = self.rle_encode(content)
        huffman_encoded, codes = self.huffman_encode(rle_encoded)
        return huffman_encoded, codes, len(content)

    def decompress_file(self, compressed_bits: str, codes: dict, original_size: int) -> str:
        try:
            huffman_decoded = self.huffman_decode(compressed_bits, codes)
            rle_decoded = self.rle_decode(huffman_decoded)
            return rle_decoded
        except Exception as e:
            raise ValueError(f"Ошибка при декомпрессии: {e}")

    def create_archive(self, archive_name, files_to_compress):
        valid_files = []
        for fp in files_to_compress:
            if not os.path.exists(fp):
                print(f"Предупреждение: Файл {fp} не существует, пропускаем")
                continue
            if not os.path.isfile(fp):
                print(f"Предупреждение: {fp} не является файлом, пропускаем")
                continue
            valid_files.append(fp)

        if not valid_files:
            raise ValueError("Нет валидных файлов для архивации")

        if len(valid_files) > 255:
            raise ValueError("Слишком много файлов (>255)")

        with open(archive_name, 'wb') as archive:
            archive.write(struct.pack('B', len(valid_files)))

            metadata_list = []
            all_bitstrings = []

            for filepath in valid_files:
                filename = os.path.basename(filepath)
                if len(filename.encode('utf-8')) > 255:
                    raise ValueError(f"Имя файла слишком длинное: {filename}")

                huffman_bits, codes, orig_size = self.compress_file(filepath)
                bit_length = len(huffman_bits)

                metadata_list.append((filename, orig_size, bit_length, codes))
                all_bitstrings.append(huffman_bits)

            for filename, orig_size, bit_length, codes in metadata_list:
                name_bytes = filename.encode('utf-8')
                archive.write(struct.pack('B', len(name_bytes)))
                archive.write(name_bytes)
                archive.write(struct.pack('I', orig_size))
                archive.write(struct.pack('I', bit_length))
                codes_bytes = pickle.dumps(codes)
                archive.write(struct.pack('I', len(codes_bytes)))
                archive.write(codes_bytes)

            for bits in all_bitstrings:
                byte_data = self.bitstring_to_bytes(bits)
                archive.write(byte_data)

    def extract_archive(self, archive_name, extract_path):
        if not os.path.exists(archive_name):
            raise FileNotFoundError(f"Архив не найден: {archive_name}")

        os.makedirs(extract_path, exist_ok=True)

        with open(archive_name, 'rb') as archive:
            num_files = struct.unpack('B', archive.read(1))[0]
            if num_files == 0:
                print("Архив пуст")
                return

            metadata_list = []
            for _ in range(num_files):
                name_len = struct.unpack('B', archive.read(1))[0]
                filename = archive.read(name_len).decode('utf-8')
                orig_size = struct.unpack('I', archive.read(4))[0]
                bit_length = struct.unpack('I', archive.read(4))[0]
                codes_len = struct.unpack('I', archive.read(4))[0]
                codes = pickle.loads(archive.read(codes_len))
                metadata_list.append((filename, orig_size, bit_length, codes))

            remaining_data = archive.read()

            bit_offset = 0
            for filename, orig_size, bit_length, codes in metadata_list:
                byte_length = (bit_length + 7) // 8
                if bit_offset + byte_length > len(remaining_data):
                    raise ValueError(f"Недостаточно данных для {filename}")

                byte_chunk = remaining_data[bit_offset:bit_offset + byte_length]
                full_bitstring = self.bytes_to_bitstring(byte_chunk)
                actual_bits = full_bitstring[:bit_length]

                decompressed = self.decompress_file(actual_bits, codes, orig_size)

                output_path = os.path.join(extract_path, filename)
                output_path = os.path.abspath(output_path)
                if not output_path.startswith(os.path.abspath(extract_path)):
                    raise ValueError(f"Недопустимое имя файла: {filename}")

                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(decompressed)
                print(f"Извлечён: {output_path}")

                bit_offset += byte_length


if __name__ == "__main__":
    archiver = SimpleArchiver()

    with open("test1.txt", "w", encoding='utf-8') as f:
        f.write("AAAABBBCCCDDDEEEEEFFFFF")

    with open("test2.txt", "w", encoding='utf-8') as f:
        f.write("Hello world! " * 10)

    try:
        archiver.create_archive("my_archive.sa", ["test1.txt", "test2.txt"])
        print("Архив создан!")
    except Exception as e:
        print(f"Ошибка при создании архива: {e}")
        exit(1)

    try:
        archiver.extract_archive("my_archive.sa", "./extracted")
        print("Архив распакован!")
    except Exception as e:
        print(f"Ошибка при распаковке: {e}")
        exit(1)

    for fname in ["test1.txt", "test2.txt"]:
        with open(fname, encoding='utf-8') as f1, open(f"./extracted/{fname}", encoding='utf-8') as f2:
            orig = f1.read()
            extracted = f2.read()
            if orig != extracted:
                print(f"❌ Файл {fname} НЕ совпадает!")
                print(f"Оригинал ({len(orig)}): {repr(orig)}")
                print(f"Извлечённый ({len(extracted)}): {repr(extracted)}")
                exit(1)
    print("✅ Все файлы совпадают!")

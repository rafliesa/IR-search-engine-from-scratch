import array


class StandardPostings:
    """
    Provides static methods to convert a postings list (list of integers)
    into a raw byte sequence using Python's built-in array library.

    Assumption: the postings list for a single term fits in memory.

    See also:
        https://docs.python.org/3/library/array.html
    """

    @staticmethod
    def encode(postings_list):
        """
        Encode a postings list into a stream of bytes.

        Parameters
        ----------
        postings_list : List[int]
            List of document IDs.

        Returns
        -------
        bytes
            Bytearray representing the sequence of integers in postings_list.
        """
        return array.array("L", postings_list).tobytes()

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decode a postings list from a stream of bytes.

        Parameters
        ----------
        encoded_postings_list : bytes
            Bytearray produced by :meth:`encode`.

        Returns
        -------
        List[int]
            List of document IDs decoded from encoded_postings_list.
        """
        decoded_postings_list = array.array("L")
        decoded_postings_list.frombytes(encoded_postings_list)
        return decoded_postings_list.tolist()

    @staticmethod
    def encode_tf(tf_list):
        """
        Encode a list of term frequencies into a stream of bytes.

        Parameters
        ----------
        tf_list : List[int]
            Raw term frequency values.

        Returns
        -------
        bytes
            Bytearray representing the raw term frequency of each document
            in the postings list.
        """
        return StandardPostings.encode(tf_list)

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Decode a list of term frequencies from a stream of bytes.

        Parameters
        ----------
        encoded_tf_list : bytes
            Bytearray produced by :meth:`encode_tf`.

        Returns
        -------
        List[int]
            Decoded list of term frequencies.
        """
        return StandardPostings.decode(encoded_tf_list)


class VBEPostings:
    """
    Unlike StandardPostings, which stores the raw sequence of integers,
    VBEPostings first converts the postings list to gap representation
    (delta encoding) and then compresses it with Variable-Byte Encoding.

    Example:
        [34, 67, 89, 454] → gaps → [34, 33, 22, 365] → VBE bytestream.

    Assumption: the postings list for a single term fits in memory.
    """

    @staticmethod
    def vb_encode_number(number):
        """
        Encode a single number using Variable-Byte Encoding.
        """
        bytes = []
        while True:
            bytes.insert(0, number % 128)  # prepend to the front
            if number < 128:
                break
            number = number // 128
        bytes[-1] += 128  # set the high bit of the last byte to 1
        return array.array("B", bytes).tobytes()

    @staticmethod
    def vb_encode(list_of_numbers):
        """
        Encode a list of numbers using Variable-Byte Encoding.
        """
        bytes = []
        for number in list_of_numbers:
            bytes.append(VBEPostings.vb_encode_number(number))
        return b"".join(bytes)

    @staticmethod
    def encode(postings_list):
        """
        Encode a postings list into a bytestream using Variable-Byte Encoding.

        Converts to gap representation first before encoding.

        Parameters
        ----------
        postings_list : List[int]
            Sorted list of document IDs.

        Returns
        -------
        bytes
            VBE-compressed bytestream of the gap-encoded postings list.
        """
        gap_postings_list = [postings_list[0]]
        for i in range(1, len(postings_list)):
            gap_postings_list.append(postings_list[i] - postings_list[i - 1])
        return VBEPostings.vb_encode(gap_postings_list)

    @staticmethod
    def encode_tf(tf_list):
        """
        Encode a list of term frequencies into a bytestream using Variable-Byte Encoding.

        Parameters
        ----------
        tf_list : List[int]
            Raw term frequency values.

        Returns
        -------
        bytes
            VBE-compressed bytestream of the term frequency list.
        """
        return VBEPostings.vb_encode(tf_list)

    @staticmethod
    def vb_decode(encoded_bytestream):
        """
        Decode a bytestream previously encoded with Variable-Byte Encoding.
        """
        n = 0
        numbers = []
        decoded_bytestream = array.array("B")
        decoded_bytestream.frombytes(encoded_bytestream)
        bytestream = decoded_bytestream.tolist()
        for byte in bytestream:
            if byte < 128:
                n = 128 * n + byte
            else:
                n = 128 * n + (byte - 128)
                numbers.append(n)
                n = 0
        return numbers

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decode a postings list from a VBE-compressed bytestream.

        Reverses gap encoding after VBE decoding to reconstruct original docIDs.

        Parameters
        ----------
        encoded_postings_list : bytes
            Bytestream produced by :meth:`encode`.

        Returns
        -------
        List[int]
            Reconstructed sorted list of document IDs.
        """
        decoded_postings_list = VBEPostings.vb_decode(encoded_postings_list)
        total = decoded_postings_list[0]
        ori_postings_list = [total]
        for i in range(1, len(decoded_postings_list)):
            total += decoded_postings_list[i]
            ori_postings_list.append(total)
        return ori_postings_list

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Decode a list of term frequencies from a VBE-compressed bytestream.

        Parameters
        ----------
        encoded_tf_list : bytes
            Bytestream produced by :meth:`encode_tf`.

        Returns
        -------
        List[int]
            Decoded list of term frequencies.
        """
        return VBEPostings.vb_decode(encoded_tf_list)


class EliasGammaPostings:
    """
    Encodes and decodes postings lists and term frequency lists using
    Elias Gamma coding over gap-encoded sequences.

    Postings are first converted to gap representation (delta encoding)
    before being compressed bit-by-bit into a bytestream. A 3-bit
    padding header is prepended so the decoder can correctly strip
    trailing padding bits.

    Assumption: the postings list for a single term fits in memory.
    """

    @staticmethod
    def elias_gamma_encode_number(number):
        """
        Encode a single positive integer using Elias Gamma coding.

        Parameters
        ----------
        number : int
            A positive integer to encode.

        Returns
        -------
        List[int]
            Bit sequence representing the Elias Gamma code of *number*.
        """
        number += 1
        binary = bin(number)[2:]
        length = len(binary)
        coded = "0" * (length - 1) + binary
        return [int(b) for b in coded]

    @staticmethod
    def elias_gamma_encode(list_of_numbers):
        """
        Encode a list of positive integers into a bytestream using Elias
        Gamma coding.

        A 3-bit header is written at the start of the stream to store the
        number of padding bits appended at the end.

        Parameters
        ----------
        list_of_numbers : List[int]
            Positive integers to encode.

        Returns
        -------
        bytes
            Packed bytestream with a 3-bit padding header.
        """
        all_bytes = []
        for number in list_of_numbers:
            all_bytes.extend(EliasGammaPostings.elias_gamma_encode_number(number))

        padding = (8 - ((len(all_bytes) + 3) % 8)) % 8
        header_bits = [(padding >> 2) & 1, (padding >> 1) & 1, padding & 1]
        all_bits = header_bits + all_bytes + [0] * padding

        result = bytearray()
        for i in range(0, len(all_bits), 8):
            byte = all_bits[i : i + 8]

            bit_string = "".join(map(str, byte))
            byte_value = int(bit_string, 2)

            result.append(byte_value)

        return bytes(result)

    @staticmethod
    def encode(postings_list):
        """
        Encode a postings list into a compressed bytestream.

        Converts to gap representation first, then applies Elias Gamma
        coding via :meth:`elias_gamma_encode`.

        Parameters
        ----------
        postings_list : List[int]
            Sorted list of document IDs.

        Returns
        -------
        bytes
            Compressed bytestream of the gap-encoded postings list.
        """
        gap_postings_list = [postings_list[0]]
        for i in range(1, len(postings_list)):
            gap_postings_list.append(postings_list[i] - postings_list[i - 1])
        return EliasGammaPostings.elias_gamma_encode(gap_postings_list)

    @staticmethod
    def encode_tf(tf_list):
        """
        Encode a list of term frequencies into a compressed bytestream.

        Parameters
        ----------
        tf_list : List[int]
            Raw term frequency values.

        Returns
        -------
        bytes
            Compressed bytestream of the term frequency list.
        """
        return EliasGammaPostings.elias_gamma_encode(tf_list)

    @staticmethod
    def elias_gamma_decode(encoded_bytestream):
        """
        Decode a bytestream produced by :meth:`elias_gamma_encode`.

        Reads the 3-bit padding header, strips padding bits, then
        iteratively decodes each Elias Gamma code word.

        Parameters
        ----------
        encoded_bytestream : bytes
            Bytestream previously encoded with :meth:`elias_gamma_encode`.

        Returns
        -------
        List[int]
            Decoded list of positive integers.
        """
        all_bits = []
        for byte in encoded_bytestream:
            to_bits = [int(bit) for bit in format(byte, "08b")]
            all_bits.extend(to_bits)

        padding = all_bits[0] * 4 + all_bits[1] * 2 + all_bits[2]
        if padding > 0:
            data_bits = all_bits[3 : len(all_bits) - padding]
        else:
            data_bits = all_bits[3:]

        numbers = []
        pos = 0

        while pos < len(data_bits):
            k = 0
            while pos < len(data_bits) and data_bits[pos] == 0:
                k += 1
                pos = pos + 1

            if pos >= len(data_bits):
                break

            bits = data_bits[pos : pos + k + 1]
            bit_string = "".join(map(str, bits))
            numbers.append(int(bit_string, 2) - 1)  # Shift -1 to reverse the +1 in encode
            pos = pos + k + 1

        return numbers

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decode a compressed postings list bytestream.

        Reverses gap encoding after applying :meth:`elias_gamma_decode`
        to reconstruct the original document IDs.

        Parameters
        ----------
        encoded_postings_list : bytes
            Bytestream produced by :meth:`encode`.

        Returns
        -------
        List[int]
            Reconstructed sorted list of document IDs.
        """
        decoded_postings_list = EliasGammaPostings.elias_gamma_decode(
            encoded_postings_list
        )

        total = decoded_postings_list[0]
        ori_postings_list = [total]
        for i in range(1, len(decoded_postings_list)):
            total += decoded_postings_list[i]
            ori_postings_list.append(total)
        return ori_postings_list

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Decode a compressed term frequency bytestream.

        Parameters
        ----------
        encoded_tf_list : bytes
            Bytestream produced by :meth:`encode_tf`.

        Returns
        -------
        List[int]
            Decoded list of term frequencies.
        """
        return EliasGammaPostings.elias_gamma_decode(encoded_tf_list)


if __name__ == "__main__":
    postings_list = [34, 67, 89, 454, 2345738]
    tf_list = [12, 10, 3, 4, 1]

    for Postings in [StandardPostings, VBEPostings, EliasGammaPostings]:
        print(Postings.__name__)
        encoded_postings_list = Postings.encode(postings_list)
        encoded_tf_list = Postings.encode_tf(tf_list)
        print("encoded postings bytes : ", encoded_postings_list)
        print("encoded postings size  : ", len(encoded_postings_list), "bytes")
        print("encoded TF list bytes  : ", encoded_tf_list)
        print("encoded TF list size   : ", len(encoded_tf_list), "bytes")

        decoded_posting_list = Postings.decode(encoded_postings_list)
        decoded_tf_list = Postings.decode_tf(encoded_tf_list)
        print("decoded postings : ", decoded_posting_list)
        print("decoded TF list  : ", decoded_tf_list)
        assert decoded_posting_list == postings_list, (
            "decoded postings do not match the original"
        )
        assert decoded_tf_list == tf_list, (
            "decoded TF list does not match the original"
        )
        print()

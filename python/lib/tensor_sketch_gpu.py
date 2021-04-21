# GPU TENSOR SKETCH

from numba import cuda

from lib.base import *
from lib.tensor_sketch import TS

# CUDA kernel to sketch a list of sequences.
# A, t, D, L (int32): parameters as usual.
# global_hashes (int32[:, :]): A*t device array of hashes.
# global_signs (float32[:, :]): A*t device array of signs. Note that these are
# floats to avoid additional (slow) int32->float32 conversions.
# seq (int8[:]): concatenation of the sequences to sketch.
# starts (int32[:]): the start positions of the subsequences in seq.
# T: (float32[:, :]): n*D device array for the output, given n input sequences.
@cuda.jit(fastmath=True)
def _gpu_sketch(A, t, D, L, hashes, signs, seq, starts, T):
    seqid = cuda.blockIdx.x
    start = starts[seqid]
    end = starts[seqid + 1]

    l = cuda.threadIdx.x
    k = cuda.threadIdx.y
    assert k < t
    assert l < D // L

    # We use a 2*(t+1)*D tensor consisting of two 'planes'.
    # At each step, one plane is the input, and one is the output. Which is indicated by `j` further down.
    plane = (t + 1) * D
    threads = t * D // L

    # Slice the shared memory into local shared memory arrays.
    # Note the different types per view.

    # NOTE: Tin has a variable offset of k*D to save a bit on further computations.
    Tin = cuda.shared.array(shape=0, dtype=nb.float32)[k * D : 2 * plane]
    local_seq = cuda.shared.array(shape=0, dtype=nb.int32)[2 * plane : 2 * plane + threads]

    local_signs = cuda.shared.array(shape=0, dtype=nb.float32)[
        2 * plane + threads : 2 * plane + threads + A * t
    ]
    local_hashes = cuda.shared.array(shape=0, dtype=nb.int32)[
        2 * plane + threads + A * t : 2 * plane + threads + 2 * A * t
    ]

    # Copy the device memory hashes/signs to shared memory.
    if l < A:
        local_hashes[l * t + k] = hashes[l][k]
        local_signs[l * t + k] = signs[l][k]

    # Initialize the tensors to 0.
    for ll in range(l, D, D // L):
        Tin[0 * plane + 0 * D + ll] = 0
        Tin[0 * plane + (0 + 1) * D + ll] = 0
        Tin[1 * plane + 0 * D + ll] = 0
        Tin[1 * plane + (0 + 1) * D + ll] = 0

    cuda.syncthreads()

    # Initialize the 0-element of the tensor to 1.
    if k == 0:
        Tin[0] = 1
        Tin[plane] = 1

    cuda.syncthreads()

    # The offset for the plane we're currently reading from. The write offset
    # is the other plane: `plane-read_plane`.
    read_plane = 0

    # Loop over characters in the sequence.
    tid = l + k * D // L
    for i in range((end - start) // threads):
        # Read `threads` characters from `seq` and store them in `local_seq` in shared memory.
        idx = start + i * threads + tid
        local_seq[tid] = seq[idx]
        cuda.syncthreads()

        # Process the fetched characters.
        for c in local_seq:
            h = local_hashes[c * t + k]
            s = local_signs[c * t + k]
            write_plane = plane - read_plane
            # Process L consecutive indices (of the D in total).
            # 0 <= l < D/L, so this covers all of [0, D).
            for ll in range(L * l, L * (l + 1)):
                # Compute the shifted target index, avoiding a modulo operation.
                r = ll + h
                r -= D if r >= D else 0
                # Write to output tensor.
                Tin[write_plane + D + ll] = Tin[read_plane + D + ll] + s * Tin[read_plane + r]

            # After this thread has processed the current character `c`, swap the active plane and wait for other threads.
            read_plane = write_plane
            cuda.syncthreads()

    # Process the remaining characters. We don't do synchronous prefetching to
    # shared memory here, because this only covers the last few characters of
    # the sequence.
    # TODO: If sequences are short, it may actually be beneficial to still do this.
    for idx in range(start + (end - start) // threads * threads, end):
        c = seq[idx]
        # Same code as above.
        h = local_hashes[c * t + k]
        s = local_signs[c * t + k]
        write_plane = plane - read_plane
        for ll in range(L * l, L * (l + 1)):
            r = ll + h
            r -= D if r >= D else 0
            Tin[write_plane + D + ll] = Tin[read_plane + D + ll] + s * Tin[read_plane + r]

        read_plane = write_plane
        cuda.syncthreads()

    # Copy to result.
    for ll in range(l, D, D // L):
        T[seqid][k][ll] = Tin[read_plane + ll]
        T[seqid][k + 1][ll] = Tin[read_plane + D + ll]


class GTS(Sketcher):
    def __init__(self, params):
        super().__init__(params)

        # Use the jitclass TS to copy hashes and signs parameters.
        # This is needed, because calling random returns different random
        # numbers inside and outside of jitted functions.
        # Ideally we'd inherit from TS, but inheriting from jitted classes is
        # not possible.
        self.ts = TS(params)
        self.hashes = np.array(self.ts.hashes, dtype=np.int32)
        self.signs = np.array(self.ts.signs, dtype=np.float32)

        self.d_hashes = cuda.to_device(self.hashes)
        self.d_signs = cuda.to_device(self.signs)

    def sketch(self, seqs: list[Sequence]) -> list[SketchedSequence]:
        assert isinstance(seqs, List)
        assert len(seqs) > 0
        assert isinstance(seqs[0], Sequence)

        # TODO: Add normalization to the GPU sketch method.
        for seq in seqs:
            assert (
                seq.len() ** self.t < 10 ** 38
            ), "Counts may overflow! Lower t or shorten the sequence."

        # Sort by decreasing length
        seqs = sorted(seqs, key=lambda s: len(s.seq), reverse=True)

        # Put all operations on a stream, so that the python code runs asynchronously of the GPU code.
        stream = cuda.stream()

        # Launch one thread block per sequence.
        blocks = len(seqs)

        # Convert the input sequences to a single list of characters and the corresponding start indices.
        raw_seqs = [seq.seq for seq in seqs]
        raw_seq = np.concatenate(raw_seqs)
        starts = np.array(
            np.cumsum(np.array([0] + [len(seq) for seq in raw_seqs]), dtype=np.int32),
            dtype=np.int32,
        )

        # Copy data from host to device.
        d_raw_seq = cuda.to_device(raw_seq, stream=stream)
        d_starts = cuda.to_device(starts, stream=stream)
        d_T = cuda.device_array((blocks, self.t + 1, self.D), dtype=np.float32, stream=stream)

        threads = self.t * self.D // self.L

        # Make sure we have enough threads to initialize self.hashes and
        # self.signs by a single synchronous copy.
        assert self.DL >= self.A

        # One thread per (l, k) <= (D/L, t)
        _gpu_sketch[
            (blocks, 1),
            (self.DL, self.t),
            stream,
            4 * (threads + 2 * (self.t + 1) * self.D + 2 * self.A * self.t),
        ](
            np.int32(self.A),
            np.int32(self.t),
            np.int32(self.D),
            np.int32(self.L),
            self.d_hashes,
            self.d_signs,
            d_raw_seq,
            d_starts,
            d_T,
        )

        T = d_T.copy_to_host(stream=stream)

        # Only return the length t sketch
        sketched_seqs = List()
        for seq, sketch in zip(seqs, T):
            self.ts._normalize(seq, sketch[self.t])
            sketched_seqs.append(SketchedSequence(seq, sketch[self.t]))

        return sketched_seqs

    def sketch_one(self, seq: Sequence) -> SketchedSequence:
        return self.sketch(List([seq]))[0]

# TENSOR SKETCH

from lib.base import *


@jitclass(sketchparams_spec + [('hashes', nb.int32[:, :]), ('signs', nb.float32[:, :])])
class TS(Sketcher):
    __init__Sketcher = Sketcher.__init__

    def __init__(self, params):
        self.__init__Sketcher(params)

        random.seed(31415)
        # An A*t array of random integers in [0, D)
        self.hashes = np.empty((self.A, self.t), dtype=np.int32)
        # An A*t array of random +-1
        self.signs = np.empty((self.A, self.t), dtype=np.float32)
        for c in range(self.A):
            for k in range(self.t):
                self.hashes[c][k] = random.randrange(0, self.D)
                self.signs[c][k] = random.randrange(-1, 2, 2)

    def _full_sketch(self, seq):
        # NOTE: The sketch is stored as float64 here so counting won't overflow.
        T = np.zeros((self.t + 1, self.D), dtype=np.float64)
        T[0][0] = 1

        for c in seq.seq:
            for k in range(self.t - 1, -1, -1):
                h = self.hashes[c][k]
                s = self.signs[c][k]
                for l in range(self.D):
                    r = l + h if l + h < self.D else l + h - self.D
                    T[k + 1][l] += s * T[k][r]

        return T

    def _normalize(self, seq, T):
        if self.normalize:
            # Normalization factor.
            n = seq.len()
            nct = nb.float64(1)
            for i in range(self.t):
                nct = nct * (n - i) / (i + 1)
            T /= nct
        return T

    def sketch_one(self, seq: Sequence) -> SketchedSequence:
        full_sketch = self._full_sketch(seq)

        self._normalize(seq, full_sketch[self.t])

        sketch = np.array([x for x in full_sketch[self.t]], dtype=nb.float32)
        return SketchedSequence(seq, sketch)

    def sketch(self, seqs: list[Sequence]) -> list[SketchedSequence]:
        return [self.sketch_one(seq) for seq in seqs]

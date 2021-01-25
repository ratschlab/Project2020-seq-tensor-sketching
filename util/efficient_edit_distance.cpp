//
// Created by amir on 31/12/2020.
//

#include <iostream>
#include <vector>
#include <ctime>

typedef char char_type;
typedef uint32_t dist_t;
typedef uint32_t result_t;

#define MIN(x,y) ((x) < (y) ? (x) : (y)) //calculate minimum between two values


dist_t edit_distance(char *s1, char *s2, unsigned int l1, unsigned int l2) {
    std::vector<std::vector<unsigned int>> dist(l2+1, std::vector<unsigned int>(l1+1, (l1+l2)*2));
    for(size_t i=0;i<=l1;i++) {
        dist[0][i] = i;
    }
    for(size_t j=0;j<=l2;j++) {
        dist[j][0] = j;
    }
    for (size_t j=1;j<=l1;j++) {
        for(size_t i=1;i<=l2;i++) {
            size_t track, t;
            if(s1[i-1] == s2[j-1]) {
                track= 0;
            } else {
                track = 1;
            }
            t = MIN((dist[i-1][j]+1),(dist[i][j-1]+1));
            dist[i][j] = MIN(t,(dist[i-1][j-1]+track));
        }
    }
    return dist[l2][l1];
}

dist_t edit_distance_v2(char *s1, char *s2, unsigned int l1, unsigned int l2) {
    // init dist = (l2+1)x(l1+1)
    std::vector<std::vector<unsigned int>> dist(l2+1, std::vector<unsigned int>(l1+1, (l1+l2)*2));
    for(size_t i=0;i <= l1;i++) {
        dist[0][i] = i;
    }
    for(size_t j=0;j <= l2;j++) {
        dist[j][0] = j;
    }
    for (size_t j=1;j <= l1;j++) {
        for(size_t i=1;i <= l2;i++) {
            size_t track = s1[i-1] != s2[j-1];
            size_t t = MIN((dist[i-1][j]+1),(dist[i][j-1]+1));
            dist[i][j] = MIN(t,(dist[i-1][j-1]+track));
        }
    }
    return dist[l2][l1];
}

dist_t edit_distance_lowmem(char *s1, char *s2, unsigned int l1, unsigned int l2) {
    // init dist = (l2+1)x(l1+1)
    dist_t *mem, *d0, *d1, *tmp;
    mem = (dist_t *) malloc((l1+1) * sizeof(dist_t) * 2);
    d0 = mem;
    d1 = mem + (l1+1);
    for(size_t i=0;i<=l1;i++) {
        d0[i] = i;
    }
    for(size_t i=1;i<=l2;i++) { // d1 = row dist[i,:], d0 = row  [i-1,:]
        d0[0] = i-1;
        d1[0] = i;
        for (size_t j=1;j<=l1;j++) {
            size_t track, t;
            track = s1[i-1] != s2[j-1];
            t = MIN((d0[j]+1),(d1[j-1]+1));
            d1[j] = MIN(t,(d0[j-1]+track));
        }
        std::swap(d0, d1);
    }
    auto ed =  d1[l1];
    free(mem);
    return ed;
}


dist_t edit_distance_lowmem_vec(char *s1, char *s2, unsigned int l1, unsigned int l2) {
    std::vector<dist_t> d0(l1+1), d1(l1+1);
    for(size_t i=0;i<=l1;i++) {
        d0[i] = i;
    }
    for(size_t i=1;i<=l2;i++) { // d1 = row dist[i,:], d0 = row  [i-1,:]
        d0[0] = i-1;
        d1[0] = i;
        for (size_t j=1;j<=l1;j++) {
            size_t track, t;
            track = s1[i-1] != s2[j-1];
            t = MIN((d0[j]+1),(d1[j-1]+1));
            d1[j] = MIN(t,(d0[j-1]+track));
        }
        std::swap(d0, d1);
    }

    auto ed =  d1[l1];
    return ed;
}


// only for equal length sequences
dist_t edit_distance_diagonal(char *seq1, char *seq2, unsigned int len) {

    dist_t *tmp,
            *diag0 = new dist_t [len + 1],
            *diag1 = new dist_t [len + 1],
            *diag2 = new dist_t [len + 1];
    for (unsigned int l = 0; l<=len; l++) {
        for (unsigned int i=0; i<=l; i++) {
            if (i<=0 or i>=l) {
                diag2[i] = l;
            } else {
                result_t t = seq1[i-1]!=seq2[l-i-1];
                diag2[i] = MIN(t + diag0[i - 1], MIN(diag1[i], diag1[i - 1]) + 1 );
            }
        }
        // rotate pointers to diagonal vectors
        tmp = diag0; diag0 = diag1; diag1 = diag2; diag2 = tmp;
    }
    for (unsigned int d = len+1; d <= 2*len; d++) { //
        int off = d - len, off2 = (d != len + 1);// diagonal: d
        for (unsigned int i = 0; i <= 2 * len - d; i++) {
            result_t t = seq1[(off + i) - 1] != seq2[d - (i + off) - 1];
            diag2[i] = MIN(t + diag0[i + off2], MIN(diag1[i], diag1[i + 1]) + 1);
        }
        // rotate pointers to diagonal vectors
        tmp = diag0; diag0 = diag1; diag1 = diag2; diag2 = tmp;
    }
    auto ed = diag1[0];
    free(diag0);
    free(diag1);
    free(diag2);
    return ed;
}

// pad the shorter sequence to become equal length
dist_t edit_distance_diagonal(char *seq1, char *seq2, unsigned int len, unsigned int len2) {
    dist_t ed;
    if (len<len2) {
        char* mem = (char*) calloc((len2+1) ,  sizeof (char));
        for (unsigned int i=0; i<len; i++)
            mem[i] = seq1[i];
        ed = edit_distance_diagonal(mem, seq2, len2);
        free(mem);
    } else if (len2 < len ){
        char* mem = (char*) calloc((len+1) ,  sizeof (char));
        for (unsigned int i=0; i<len2; i++)
            mem[i] = seq2[i];
        ed = edit_distance_diagonal(seq1, mem, len);
        free(mem);
    } else {
        ed = edit_distance_diagonal(seq1, seq2, len);
    }
    return ed;
}


dist_t edit_distance_diagonal_vec(char *seq1, char *seq2, unsigned int len, unsigned int len2) {
    std::vector<dist_t> tmp, d0(len+1), d1(len+1), d2(len+1);
    for (unsigned int d = 0; d <= len; d++) { // diagonal: d
        for (unsigned int i=0; i <= d; i++) {
            if (i<=0 or i >= d) {
                d2[i] = d;
            } else {
                result_t t = seq1[i-1]!=seq2[d - i - 1];
                d2[i] = MIN(t + d0[i-1], MIN(d1[i],d1[i-1])+1 );
            }
        }

        tmp = std::move(d0);
        d0 = std::move(d1);
        d1 = std::move(d2);
        d2 = std::move(tmp);
    }
    for (unsigned int d = len+1; d <= 2*len; d++) { //
        int off = d-len, off2 = (d!=len+1);// diagonal: d
        for (unsigned int i=0; i <= 2* len-d; i++) {
            result_t t = seq1[(off+i)-1]!=seq2[d - (i+off) - 1];
            d2[i] = MIN(t + d0[i+off2], MIN(d1[i],d1[i+1])+1 );
        }

        tmp = std::move(d0);
        d0 = std::move(d1);
        d1 = std::move(d2);
        d2 = std::move(tmp);
    }
    return d1[0];
}

void record_time(const char* name, dist_t (*fptr)(char_type *, char_type*, unsigned, unsigned ),
                 char_type *s1, char_type *s2, unsigned int l1, unsigned int l2) {
    auto c_start = std::clock();
    fptr(s1, s2, l1, l2);
    auto time = 1.0 * (std::clock()-c_start) / CLOCKS_PER_SEC;
    printf("%s time: %f s\n", name, time);
}

bool cmp_methods (const char* name, dist_t (*fptr)(char_type *, char_type*, unsigned, unsigned ),
                  const char* name2, dist_t (*fptr2)(char_type *, char_type*, unsigned, unsigned ),
                  char_type *s1, char_type *s2, unsigned int l1, unsigned int l2) {
    auto ed1 = fptr(s1, s2, l1, l2);
    auto ed2 = fptr2(s1, s2, l1, l2);
    printf("%s: %s == %s\n", (ed1==ed2) ? "PASS" : "FAIL", name, name2);
    return (ed1==ed2);
}

void compare_times(char_type *h1, char_type *h2, unsigned int len) {

    record_time("edit_distance", &edit_distance, h1, h2, len, len);
    record_time("edit_distance_v2", &edit_distance_v2, h1, h2, len, len);
    record_time("edit_distance_lowmem", &edit_distance_lowmem, h1, h2, len, len);
    record_time("edit_distance_lowmem_vec", &edit_distance_lowmem_vec, h1, h2, len, len);
    record_time("edit_distance_diagonal_vec", &edit_distance_diagonal_vec, h1, h2, len, len);
    record_time("edit_distance_diagonal", &edit_distance_diagonal, h1, h2, len, len);
}


void test_equality(char_type *h1, char_type *h2, unsigned int len) {
    int num = 0;
    num += cmp_methods("edit_distance", &edit_distance, "edit_distance_v2", &edit_distance_v2, h1, h2, len, len);
    num += cmp_methods("edit_distance", &edit_distance, "edit_distance_lowmem_vec", &edit_distance_lowmem_vec, h1, h2, len, len);
    num += cmp_methods("edit_distance", &edit_distance, "edit_distance_lowmem", &edit_distance_lowmem, h1, h2, len, len);
    num += cmp_methods("edit_distance", &edit_distance, "edit_distance_diagonal", &edit_distance_diagonal, h1, h2, len, len);
    num += cmp_methods("edit_distance", &edit_distance, "edit_distance_diagonal_vec", &edit_distance_diagonal_vec, h1, h2, len, len);
    printf("%d out of %d passed!!\n", num, 5);
}


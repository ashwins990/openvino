// Copyright (C) 2024 FUJITSU LIMITED
// SPDX-License-Identifier: Apache-2.0
//
#include <arm_sve.h>
#define SIZE_IN_BITS(t_var) sizeof(t_var) * 8
#define __ce(expr, bits, ...) if constexpr(expr == bits){ __VA_ARGS__}

#define SVE_PREDICATE(var, t_var)   \
    svbool_t var;                   \
                                    \
    __ce(SIZE_IN_BITS(t_var),  8,   \
            var = svptrue_b8();     \
    )                               \
    __ce(SIZE_IN_BITS(t_var), 16,   \
            var = svptrue_b16();    \
    )                               \
    __ce(SIZE_IN_BITS(t_var), 32,   \
            var = svptrue_b32();    \
    )                               \
    __ce(SIZE_IN_BITS(t_var), 64,   \
            var = svptrue_b64();    \
    )

#define SVE_VLEN(var, t_var)        \
    size_t var;                     \
                                    \
    __ce(SIZE_IN_BITS(t_var),  8,   \
            var = svcntb();         \
    )                               \
    __ce(SIZE_IN_BITS(t_var), 16,   \
            var = svcnth();         \
    )                               \
    __ce(SIZE_IN_BITS(t_var), 32,   \
            var = svcntw();         \
    )                               \
    __ce(SIZE_IN_BITS(t_var), 64,   \
            var = svcntd();         \
    )


static void cvt_copy(float* dst, float16_t* src, int n){
        auto pg_vl2   = svwhilelt_b16(svcnth() / 2, svcnth());
        auto vlen     = svcnth() / 2;
        auto scratch  = svdup_f16_x(svptrue_b16(), 0);
        auto pg_dst   = svptrue_b32();
        int i = 0;
        for(; i + vlen <= n; i += vlen) {
                auto load_src       = svld1_f16(pg_vl2, src + i);
                auto src_interleave = svzip1_f16(load_src, scratch);
                auto cvt_dst        = svcvt_f32_f16_z(pg_dst, src_interleave);
                svst1(pg_dst, dst + i, cvt_dst);
        }
        for (; i < n; i++) {
                dst[i] = src[i];
        }   
}

static void cvt_copy(float16_t* dst, float* src, int n){
        auto pg_src = svptrue_b32();
        auto pg_dst = svwhilelt_b16(svcnth() / 2, svcnth());
        auto vlen   = svcntw();
        auto scratch  = svdup_f16_x(svptrue_b16(), 0);
        int i = 0;
        for (; i + vlen < n; i += vlen){
                auto load_src = svld1_f32(pg_src, src + i);
                auto cvt_dst  = svcvt_f16_f32_z(pg_src, load_src);
                auto str_dst  = svuzp1(cvt_dst, scratch);
                svst1_f16(pg_dst, dst + i, str_dst);
        }
        for (; i < n; i++) {
                dst[i] = src[i];
        }  

}

void inline gemm_acl_ref(float16_t* a, float16_t* b, float16_t* c, size_t m, size_t n, size_t k, size_t lda, size_t ldb, size_t ldc, bool acc){
    size_t vlen = svcnth();
    auto pg = svptrue_b16();
    
    for(size_t i = 0; i < m; ++i ){

        for(size_t j = 0; j < n ; j += 1){

            auto vsum = svdup_n_f16_z(pg, 0.0);

            for(size_t x = 0; x + vlen <= k ; x += vlen){

                auto a1 = svld1(pg, a + x + i * lda);
                auto b1 = svld1(pg, b + x + j * ldb);
                vsum = svmla_f16_z(pg, vsum, a1, b1);

            }
            if(acc)
                *(c + j) =  *(c + j) + svaddv_f16(pg, vsum);
            else
                *(c + j) =  svaddv_f16(pg, vsum);

        }
        c += ldc; 
    }

}
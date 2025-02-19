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

inline void gemm_1x128x32(float16_t* a, float16_t* b, float16_t* c){
        // predicate setting all true
        svbool_t pg = svptrue_b16();

        // predicate to load only 8 f16 values
        svbool_t pgVL8 = svptrue_pat_b16(SV_VL8);

        // 2 accumulators needed (32 / 16 = 2)
        svfloat16_t acc1 = svdup_n_f16_z(pg, 0.0);
        svfloat16_t acc2 = svdup_n_f16_z(pg, 0.0);

        for(int i = 0; i + 16 <= 128; i +=16 ){

                svfloat16_t ra =svld1rq_f16(pgVL8, a + i);
                svfloat16_t ra1 =svld1rq_f16(pgVL8, a + i + 8);
                
                svfloat16_t rb1  = svld1(pg, b);
                svfloat16_t rb2  = svld1(pg, b + 16 * 1);
                svfloat16_t rb3  = svld1(pg, b + 16 * 2);
                svfloat16_t rb4  = svld1(pg, b + 16 * 3);
                svfloat16_t rb5  = svld1(pg, b + 16 * 4);
                svfloat16_t rb6  = svld1(pg, b + 16 * 5);
                svfloat16_t rb7  = svld1(pg, b + 16 * 6);
                svfloat16_t rb8  = svld1(pg, b + 16 * 7);
                svfloat16_t rb9  = svld1(pg, b + 16 * 8);
                svfloat16_t rb10 = svld1(pg, b + 16 * 9);
                svfloat16_t rb11 = svld1(pg, b + 16 * 10);
                svfloat16_t rb12 = svld1(pg, b + 16 * 11);
                svfloat16_t rb13 = svld1(pg, b + 16 * 12);
                svfloat16_t rb14 = svld1(pg, b + 16 * 13);
                svfloat16_t rb15 = svld1(pg, b + 16 * 14);
                svfloat16_t rb16 = svld1(pg, b + 16 * 15);


                acc1 = svmla_lane_f16(acc1, rb1, ra, 0);
                acc2 = svmla_lane_f16(acc2, rb2, ra, 0);

                acc1 = svmla_lane_f16(acc1, rb3, ra, 1);
                acc2 = svmla_lane_f16(acc2, rb4, ra, 1);

                acc1 = svmla_lane_f16(acc1, rb5, ra, 2);
                acc2 = svmla_lane_f16(acc2, rb6, ra, 2);

                acc1 = svmla_lane_f16(acc1, rb7, ra, 3);
                acc2 = svmla_lane_f16(acc2, rb8, ra, 3);

                acc1 = svmla_lane_f16(acc1, rb9 , ra, 4);
                acc2 = svmla_lane_f16(acc2, rb10, ra,  4);

                acc1 = svmla_lane_f16(acc1, rb11, ra,  5);
                acc2 = svmla_lane_f16(acc2, rb12, ra,  5);

                acc1 = svmla_lane_f16(acc1, rb13, ra,  6);
                acc2 = svmla_lane_f16(acc2, rb14, ra,  6);

                acc1 = svmla_lane_f16(acc1, rb15, ra, 7);
                acc2 = svmla_lane_f16(acc2, rb16, ra, 7);



                b = b + 32 * 8;

                rb1  = svld1(pg, b);
                rb2  = svld1(pg, b + 16 * 1);
                rb3  = svld1(pg, b + 16 * 2);
                rb4  = svld1(pg, b + 16 * 3);
                rb5  = svld1(pg, b + 16 * 4);
                rb6  = svld1(pg, b + 16 * 5);
                rb7  = svld1(pg, b + 16 * 6);
                rb8  = svld1(pg, b + 16 * 7);
                rb9  = svld1(pg, b + 16 * 8);
                rb10 = svld1(pg, b + 16 * 9);
                rb11 = svld1(pg, b + 16 * 10);
                rb12 = svld1(pg, b + 16 * 11);
                rb13 = svld1(pg, b + 16 * 12);
                rb14 = svld1(pg, b + 16 * 13);
                rb15 = svld1(pg, b + 16 * 14);
                rb16 = svld1(pg, b + 16 * 15);


                acc1 = svmla_lane_f16(acc1, rb1, ra1, 0);
                acc2 = svmla_lane_f16(acc2, rb2, ra1, 0);

                acc1 = svmla_lane_f16(acc1, rb3, ra1, 1);
                acc2 = svmla_lane_f16(acc2, rb4, ra1, 1);

                acc1 = svmla_lane_f16(acc1, rb5, ra1, 2);
                acc2 = svmla_lane_f16(acc2, rb6, ra1, 2);

                acc1 = svmla_lane_f16(acc1, rb7, ra1, 3);
                acc2 = svmla_lane_f16(acc2, rb8, ra1, 3);

                acc1 = svmla_lane_f16(acc1, rb9 , ra1, 4);
                acc2 = svmla_lane_f16(acc2, rb10, ra1, 4);

                acc1 = svmla_lane_f16(acc1, rb11, ra1, 5);
                acc2 = svmla_lane_f16(acc2, rb12, ra1, 5);

                acc1 = svmla_lane_f16(acc1, rb13, ra1, 6);
                acc2 = svmla_lane_f16(acc2, rb14, ra1, 6);

                acc1 = svmla_lane_f16(acc1, rb15, ra1, 7);
                acc2 = svmla_lane_f16(acc2, rb16, ra1, 7);


                b = b + 32 * 8;

        }
        svst1_f16(pg, c, acc1);
        svst1_f16(pg, c + 16, acc2);

        }

inline void gemm_1x32x128(float16_t* a, float16_t* b, float16_t* c, bool acc){
        // predicate setting all true
        svbool_t pg = svptrue_b16();

        // load the values of a. 32 f16 values (32 * 2B = 64B)
        svfloat16_t rega1 = svld1_f16(pg, a);
        svfloat16_t rega2 = svld1_f16(pg, a + 16);

        // 8 accumlators required ( 128 / 16 = 8 )
        svfloat16_t acc1 ;
        svfloat16_t acc2 ;
        svfloat16_t acc3 ;
        svfloat16_t acc4 ;
        svfloat16_t acc5 ;
        svfloat16_t acc6 ;
        svfloat16_t acc7 ;
        svfloat16_t acc8 ;
        if (acc){
                acc1  = svld1_f16(pg, c);
                acc2  = svld1_f16(pg, c + 16 * 1);
                acc3  = svld1_f16(pg, c + 16 * 2);
                acc4  = svld1_f16(pg, c + 16 * 3);
                acc5  = svld1_f16(pg, c + 16 * 4);
                acc6  = svld1_f16(pg, c + 16 * 5);
                acc7  = svld1_f16(pg, c + 16 * 6);
                acc8  = svld1_f16(pg, c + 16 * 7);
        }
        else {
                acc1  = svdup_f16(0.0);
                acc2  = svdup_f16(0.0);
                acc3  = svdup_f16(0.0);
                acc4  = svdup_f16(0.0);
                acc5  = svdup_f16(0.0);
                acc6  = svdup_f16(0.0);
                acc7  = svdup_f16(0.0);
                acc8  = svdup_f16(0.0);
        }

        for(int i = 0; i + 1 <= 16; i = i + 1){
                svfloat16_t b1 = svld1_f16(pg, b);
                svfloat16_t b2 = svld1_f16(pg, b + 16 * 1);
                svfloat16_t b3 = svld1_f16(pg, b + 16 * 2);
                svfloat16_t b4 = svld1_f16(pg, b + 16 * 3);
                svfloat16_t b5 = svld1_f16(pg, b + 16 * 4);
                svfloat16_t b6 = svld1_f16(pg, b + 16 * 5);
                svfloat16_t b7 = svld1_f16(pg, b + 16 * 6);
                svfloat16_t b8 = svld1_f16(pg, b + 16 * 7);

                // svfloat16_t b9  = svld1_f16(pg, b + 16 * 8);
                // svfloat16_t b10 = svld1_f16(pg, b + 16 * 9);
                // svfloat16_t b11 = svld1_f16(pg, b + 16 * 10);
                // svfloat16_t b12 = svld1_f16(pg, b + 16 * 11);
                // svfloat16_t b13 = svld1_f16(pg, b + 16 * 12);
                // svfloat16_t b14 = svld1_f16(pg, b + 16 * 13);
                // svfloat16_t b15 = svld1_f16(pg, b + 16 * 14);
                // svfloat16_t b16 = svld1_f16(pg, b + 16 * 15);

                acc1 = svmla_f16_z(pg, acc1, b1, svdup_lane_f16(rega1, i));
                acc2 = svmla_f16_z(pg, acc2, b2, svdup_lane_f16(rega1, i));
                acc3 = svmla_f16_z(pg, acc3, b3, svdup_lane_f16(rega1, i));
                acc4 = svmla_f16_z(pg, acc4, b4, svdup_lane_f16(rega1, i));
                acc5 = svmla_f16_z(pg, acc5, b5, svdup_lane_f16(rega1, i));
                acc6 = svmla_f16_z(pg, acc6, b6, svdup_lane_f16(rega1, i));
                acc7 = svmla_f16_z(pg, acc7, b7, svdup_lane_f16(rega1, i));
                acc8 = svmla_f16_z(pg, acc8, b8, svdup_lane_f16(rega1, i));

                // acc1 = svmla_f16_z(pg, acc1, b1, svdup_lane_f16(rega1, i + 1));
                // acc2 = svmla_f16_z(pg, acc2, b1, svdup_lane_f16(rega1, i + 1));
                // acc3 = svmla_f16_z(pg, acc3, b1, svdup_lane_f16(rega1, i + 1));
                // acc4 = svmla_f16_z(pg, acc4, b1, svdup_lane_f16(rega1, i + 1));
                // acc5 = svmla_f16_z(pg, acc5, b1, svdup_lane_f16(rega1, i + 1));
                // acc6 = svmla_f16_z(pg, acc6, b1, svdup_lane_f16(rega1, i + 1));
                // acc7 = svmla_f16_z(pg, acc7, b1, svdup_lane_f16(rega1, i + 1));
                // acc8 = svmla_f16_z(pg, acc8, b1, svdup_lane_f16(rega1, i + 1));

                b = b + 1 * 128;

        }

        for(int i = 0; i + 1 <= 16; i = i + 1){
                svfloat16_t b1 = svld1_f16(pg, b);
                svfloat16_t b2 = svld1_f16(pg, b + 16 * 1);
                svfloat16_t b3 = svld1_f16(pg, b + 16 * 2);
                svfloat16_t b4 = svld1_f16(pg, b + 16 * 3);
                svfloat16_t b5 = svld1_f16(pg, b + 16 * 4);
                svfloat16_t b6 = svld1_f16(pg, b + 16 * 5);
                svfloat16_t b7 = svld1_f16(pg, b + 16 * 6);
                svfloat16_t b8 = svld1_f16(pg, b + 16 * 7);

                // svfloat16_t b9  = svld1_f16(pg, b + 16 * 8);
                // svfloat16_t b10 = svld1_f16(pg, b + 16 * 9);
                // svfloat16_t b11 = svld1_f16(pg, b + 16 * 10);
                // svfloat16_t b12 = svld1_f16(pg, b + 16 * 11);
                // svfloat16_t b13 = svld1_f16(pg, b + 16 * 12);
                // svfloat16_t b14 = svld1_f16(pg, b + 16 * 13);
                // svfloat16_t b15 = svld1_f16(pg, b + 16 * 14);
                // svfloat16_t b16 = svld1_f16(pg, b + 16 * 15);

                acc1 = svmla_f16_z(pg, acc1, b1, svdup_lane_f16(rega2, i));
                acc2 = svmla_f16_z(pg, acc2, b2, svdup_lane_f16(rega2, i));
                acc3 = svmla_f16_z(pg, acc3, b3, svdup_lane_f16(rega2, i));
                acc4 = svmla_f16_z(pg, acc4, b4, svdup_lane_f16(rega2, i));
                acc5 = svmla_f16_z(pg, acc5, b5, svdup_lane_f16(rega2, i));
                acc6 = svmla_f16_z(pg, acc6, b6, svdup_lane_f16(rega2, i));
                acc7 = svmla_f16_z(pg, acc7, b7, svdup_lane_f16(rega2, i));
                acc8 = svmla_f16_z(pg, acc8, b8, svdup_lane_f16(rega2, i));

                // acc1 = svmla_f16_z(pg, acc1, b1, svdup_lane_f16(rega2, i + 1));
                // acc2 = svmla_f16_z(pg, acc2, b1, svdup_lane_f16(rega2, i + 1));
                // acc3 = svmla_f16_z(pg, acc3, b1, svdup_lane_f16(rega2, i + 1));
                // acc4 = svmla_f16_z(pg, acc4, b1, svdup_lane_f16(rega2, i + 1));
                // acc5 = svmla_f16_z(pg, acc5, b1, svdup_lane_f16(rega2, i + 1));
                // acc6 = svmla_f16_z(pg, acc6, b1, svdup_lane_f16(rega2, i + 1));
                // acc7 = svmla_f16_z(pg, acc7, b1, svdup_lane_f16(rega2, i + 1));
                // acc8 = svmla_f16_z(pg, acc8, b1, svdup_lane_f16(rega2, i + 1));

                b = b + 1 * 128;

        }
        svst1_f16(pg, c, acc1);
        svst1_f16(pg, c + 16 * 1, acc2);
        svst1_f16(pg, c + 16 * 2, acc3);
        svst1_f16(pg, c + 16 * 3, acc4);
        svst1_f16(pg, c + 16 * 4, acc5);
        svst1_f16(pg, c + 16 * 5, acc6);
        svst1_f16(pg, c + 16 * 6, acc7);
        svst1_f16(pg, c + 16 * 7, acc8);
}

void inline gemm_qk(float16_t* a, float16_t* b, float16_t* c, size_t m, size_t n, size_t k, size_t lda, size_t ldb, size_t ldc, bool acc){

        for(size_t i = 0; i < m; ++i){
                // __builtin_prefetch(a + in_stride, 0, 3);  // Prefetch next 'a' for read
                // __builtin_prefetch(c + out_stride, 1, 3); // Prefetch next 'c' for write
                gemm_1x128x32(a, b, c);
                a = a + lda;
                c = c + ldc;
            }

}

void inline gemm_wv(float16_t* a, float16_t* b, float16_t* c, size_t m, size_t n, size_t k, size_t lda, size_t ldb, size_t ldc, bool acc){
        for(size_t i = 0; i < m; ++i){
                // __builtin_prefetch(a + in_stride, 0, 3);  // Prefetch next 'a' for read
                // __builtin_prefetch(c + out_stride, 1, 3); // Prefetch next 'c' for write
                gemm_1x32x128(a, b, c, acc);
                a = a + lda;
                c = c + ldc;
            }
}

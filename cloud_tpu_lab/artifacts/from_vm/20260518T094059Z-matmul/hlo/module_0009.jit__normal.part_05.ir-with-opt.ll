; ModuleID = '__compute_module_part_05'
source_filename = "__compute_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@multiply_convert_fusion.clone_parallel_bounds = private unnamed_addr constant [24 x [1 x [2 x i64]]] [[1 x [2 x i64]] [[2 x i64] [i64 0, i64 21]], [1 x [2 x i64]] [[2 x i64] [i64 21, i64 42]], [1 x [2 x i64]] [[2 x i64] [i64 42, i64 63]], [1 x [2 x i64]] [[2 x i64] [i64 63, i64 84]], [1 x [2 x i64]] [[2 x i64] [i64 84, i64 105]], [1 x [2 x i64]] [[2 x i64] [i64 105, i64 126]], [1 x [2 x i64]] [[2 x i64] [i64 126, i64 147]], [1 x [2 x i64]] [[2 x i64] [i64 147, i64 168]], [1 x [2 x i64]] [[2 x i64] [i64 168, i64 189]], [1 x [2 x i64]] [[2 x i64] [i64 189, i64 210]], [1 x [2 x i64]] [[2 x i64] [i64 210, i64 231]], [1 x [2 x i64]] [[2 x i64] [i64 231, i64 252]], [1 x [2 x i64]] [[2 x i64] [i64 252, i64 273]], [1 x [2 x i64]] [[2 x i64] [i64 273, i64 294]], [1 x [2 x i64]] [[2 x i64] [i64 294, i64 315]], [1 x [2 x i64]] [[2 x i64] [i64 315, i64 336]], [1 x [2 x i64]] [[2 x i64] [i64 336, i64 357]], [1 x [2 x i64]] [[2 x i64] [i64 357, i64 378]], [1 x [2 x i64]] [[2 x i64] [i64 378, i64 399]], [1 x [2 x i64]] [[2 x i64] [i64 399, i64 420]], [1 x [2 x i64]] [[2 x i64] [i64 420, i64 441]], [1 x [2 x i64]] [[2 x i64] [i64 441, i64 462]], [1 x [2 x i64]] [[2 x i64] [i64 462, i64 483]], [1 x [2 x i64]] [[2 x i64] [i64 483, i64 512]]]

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define noalias noundef ptr @multiply_convert_fusion.clone(ptr readonly captures(none) %0) local_unnamed_addr #0 {
  %workgroup_id_gep = getelementptr inbounds nuw i8, ptr %0, i64 8
  %workgroup_id = load ptr, ptr %workgroup_id_gep, align 8
  %workgroup_id_x = load i64, ptr %workgroup_id, align 4
  %args_gep = getelementptr inbounds nuw i8, ptr %0, i64 24
  %args = load ptr, ptr %args_gep, align 8
  %arg0 = load ptr, ptr %args, align 8, !invariant.load !1, !dereferenceable !2, !align !3
  %arg1_gep = getelementptr i8, ptr %args, i64 16
  %arg1 = load ptr, ptr %arg1_gep, align 8, !invariant.load !1, !dereferenceable !2, !align !3
  %arg2_gep = getelementptr i8, ptr %args, i64 32
  %arg2 = load ptr, ptr %arg2_gep, align 8, !invariant.load !1, !dereferenceable !4, !align !3
  %lo_dim_0_gep = getelementptr inbounds [24 x [1 x [2 x i64]]], ptr @multiply_convert_fusion.clone_parallel_bounds, i64 0, i64 %workgroup_id_x, i64 0, i64 0
  %up_dim_0_gep = getelementptr inbounds [24 x [1 x [2 x i64]]], ptr @multiply_convert_fusion.clone_parallel_bounds, i64 0, i64 %workgroup_id_x, i64 0, i64 1
  %lo_dim_0 = load i64, ptr %lo_dim_0_gep, align 16
  %up_dim_0 = load i64, ptr %up_dim_0_gep, align 8
  %.not2 = icmp ult i64 %lo_dim_0, %up_dim_0
  br i1 %.not2, label %vector.ph, label %return

vector.ph:                                        ; preds = %1, %multiply_convert_fusion.clone.loop_exit.dim.1
  %multiply_convert_fusion.clone.invar_address.dim.0.03 = phi i64 [ %invar.inc, %multiply_convert_fusion.clone.loop_exit.dim.1 ], [ %lo_dim_0, %1 ]
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %2 = getelementptr inbounds [512 x [512 x i32]], ptr %arg0, i64 0, i64 %multiply_convert_fusion.clone.invar_address.dim.0.03, i64 %index
  %wide.load = load <8 x i32>, ptr %2, align 32, !invariant.load !1, !noalias !5
  %3 = getelementptr inbounds [512 x [512 x i32]], ptr %arg1, i64 0, i64 %multiply_convert_fusion.clone.invar_address.dim.0.03, i64 %index
  %wide.load5 = load <8 x i32>, ptr %3, align 32, !invariant.load !1, !noalias !5
  %4 = xor <8 x i32> %wide.load5, %wide.load
  %5 = shl <8 x i32> %4, splat (i32 15)
  %6 = and <8 x i32> %5, splat (i32 8323072)
  %7 = or disjoint <8 x i32> %6, splat (i32 1065353216)
  %8 = bitcast <8 x i32> %7 to <8 x float>
  %9 = fadd <8 x float> %8, splat (float -1.000000e+00)
  %10 = bitcast <8 x float> %9 to <8 x i32>
  %11 = lshr <8 x i32> %10, splat (i32 16)
  %12 = and <8 x i32> %11, splat (i32 1)
  %13 = add <8 x i32> %10, splat (i32 32767)
  %14 = add <8 x i32> %13, %12
  %15 = and <8 x i32> %14, splat (i32 -65536)
  %16 = bitcast <8 x i32> %15 to <8 x float>
  %17 = fmul <8 x float> %16, splat (float 2.000000e+00)
  %18 = bitcast <8 x float> %17 to <8 x i32>
  %19 = lshr <8 x i32> %18, splat (i32 16)
  %20 = and <8 x i32> %19, splat (i32 1)
  %21 = fcmp uno <8 x float> %17, zeroinitializer
  %22 = and <8 x i32> %18, splat (i32 -8388608)
  %23 = or disjoint <8 x i32> %22, splat (i32 4194304)
  %24 = add <8 x i32> %18, splat (i32 32767)
  %25 = add <8 x i32> %24, %20
  %26 = and <8 x i32> %25, splat (i32 -65536)
  %27 = select <8 x i1> %21, <8 x i32> %23, <8 x i32> %26
  %28 = bitcast <8 x i32> %27 to <8 x float>
  %29 = fadd <8 x float> %28, splat (float 0xBFEFE00000000000)
  %30 = bitcast <8 x float> %29 to <8 x i32>
  %31 = lshr <8 x i32> %30, splat (i32 16)
  %32 = and <8 x i32> %31, splat (i32 1)
  %33 = fcmp uno <8 x float> %29, zeroinitializer
  %34 = and <8 x i32> %30, splat (i32 -8388608)
  %35 = or disjoint <8 x i32> %34, splat (i32 4194304)
  %36 = add <8 x i32> %30, splat (i32 32767)
  %37 = add <8 x i32> %36, %32
  %38 = and <8 x i32> %37, splat (i32 -65536)
  %39 = select <8 x i1> %33, <8 x i32> %35, <8 x i32> %38
  %40 = bitcast <8 x i32> %39 to <8 x float>
  %41 = tail call <8 x float> @llvm.maximum.v8f32(<8 x float> %40, <8 x float> splat (float 0xBFEFE00000000000))
  %42 = tail call <8 x float> @llvm.fabs.v8f32(<8 x float> %41)
  %43 = fcmp oeq <8 x float> %42, splat (float 1.000000e+00)
  %44 = fneg <8 x float> %41
  %45 = fmul <8 x float> %41, %44
  %46 = fadd <8 x float> %45, splat (float 1.000000e+00)
  %log_f32.i = fcmp ule <8 x float> %46, zeroinitializer
  %log_f321.i = sext <8 x i1> %log_f32.i to <8 x i32>
  %log_f322.i = bitcast <8 x i32> %log_f321.i to <8 x float>
  %log_f323.i = fcmp oeq <8 x float> %46, zeroinitializer
  %log_f324.i = sext <8 x i1> %log_f323.i to <8 x i32>
  %log_f325.i = bitcast <8 x i32> %log_f324.i to <8 x float>
  %log_f326.i = fcmp oeq <8 x float> %46, splat (float 0x7FF0000000000000)
  %log_f327.i = sext <8 x i1> %log_f326.i to <8 x i32>
  %log_f328.i = bitcast <8 x i32> %log_f327.i to <8 x float>
  %47 = fcmp uge <8 x float> splat (float 0x3810000000000000), %46
  %48 = select <8 x i1> %47, <8 x float> splat (float 0x3810000000000000), <8 x float> %46
  %49 = bitcast <8 x float> %48 to <8 x i32>
  %50 = lshr <8 x i32> %49, splat (i32 23)
  %log_f329.i = bitcast <8 x float> %48 to <8 x i32>
  %log_f3210.i = and <8 x i32> %log_f329.i, splat (i32 -2139095041)
  %51 = bitcast <8 x i32> %log_f3210.i to <8 x float>
  %log_f3212.i = or <8 x i32> %log_f3210.i, splat (i32 1056964608)
  %log_f3213.i = bitcast <8 x i32> %log_f3212.i to <8 x float>
  %52 = sub <8 x i32> %50, splat (i32 127)
  %53 = sitofp <8 x i32> %52 to <8 x float>
  %log_f3214.i = fadd <8 x float> splat (float 1.000000e+00), %53
  %log_f3215.i = fcmp olt <8 x float> %log_f3213.i, splat (float 0x3FE6A09E60000000)
  %log_f3216.i = sext <8 x i1> %log_f3215.i to <8 x i32>
  %log_f3217.i = bitcast <8 x i32> %log_f3216.i to <8 x float>
  %log_f3220.i = and <8 x i32> %log_f3212.i, %log_f3216.i
  %54 = bitcast <8 x i32> %log_f3220.i to <8 x float>
  %55 = fsub <8 x float> %log_f3213.i, splat (float 1.000000e+00)
  %log_f3222.i = and <8 x i32> %log_f3216.i, splat (i32 1065353216)
  %56 = bitcast <8 x i32> %log_f3222.i to <8 x float>
  %57 = fsub <8 x float> %log_f3214.i, %56
  %log_f3223.i = fadd <8 x float> %55, %54
  %log_f3224.i = fmul <8 x float> %log_f3223.i, %log_f3223.i
  %log_f3225.i = fmul <8 x float> %log_f3224.i, %log_f3223.i
  %log_f3226.i = fmul <8 x float> %log_f3223.i, splat (float 0x3FB2043760000000)
  %log_f3227.i = fadd <8 x float> splat (float 0xBFBD7A3700000000), %log_f3226.i
  %log_f3228.i = fmul <8 x float> %log_f3223.i, splat (float 0xBFBFCBA9E0000000)
  %log_f3229.i = fadd <8 x float> splat (float 0x3FC23D37E0000000), %log_f3228.i
  %log_f3230.i = fmul <8 x float> %log_f3223.i, splat (float 0x3FC999D580000000)
  %log_f3231.i = fadd <8 x float> splat (float 0xBFCFFFFF80000000), %log_f3230.i
  %log_f3232.i = fmul <8 x float> %log_f3227.i, %log_f3223.i
  %log_f3233.i = fadd <8 x float> splat (float 0x3FBDE4A340000000), %log_f3232.i
  %log_f3234.i = fmul <8 x float> %log_f3229.i, %log_f3223.i
  %log_f3235.i = fadd <8 x float> splat (float 0xBFC555CA00000000), %log_f3234.i
  %log_f3236.i = fmul <8 x float> %log_f3231.i, %log_f3223.i
  %log_f3237.i = fadd <8 x float> splat (float 0x3FD5555540000000), %log_f3236.i
  %log_f3238.i = fmul <8 x float> %log_f3233.i, %log_f3225.i
  %log_f3239.i = fadd <8 x float> %log_f3235.i, %log_f3238.i
  %log_f3240.i = fmul <8 x float> %log_f3239.i, %log_f3225.i
  %log_f3241.i = fadd <8 x float> %log_f3237.i, %log_f3240.i
  %log_f3242.i = fmul <8 x float> %log_f3241.i, %log_f3225.i
  %log_f3243.i = fmul <8 x float> splat (float 0xBF2BD01060000000), %57
  %log_f3244.i = fmul <8 x float> splat (float 5.000000e-01), %log_f3224.i
  %log_f3245.i = fadd <8 x float> %log_f3242.i, %log_f3243.i
  %58 = fsub <8 x float> %log_f3223.i, %log_f3244.i
  %log_f3246.i = fmul <8 x float> splat (float 0x3FE6300000000000), %57
  %log_f3247.i = fadd <8 x float> %58, %log_f3245.i
  %log_f3248.i = fadd <8 x float> %log_f3247.i, %log_f3246.i
  %log_f3250.i = and <8 x i32> %log_f324.i, splat (i32 -8388608)
  %59 = bitcast <8 x i32> %log_f3250.i to <8 x float>
  %log_f3252.i = and <8 x i32> %log_f327.i, splat (i32 2139095040)
  %60 = bitcast <8 x i32> %log_f3252.i to <8 x float>
  %log_f3255.i = or <8 x i32> %log_f3250.i, %log_f3252.i
  %log_f3256.i = bitcast <8 x i32> %log_f3255.i to <8 x float>
  %log_f3257.i = bitcast <8 x float> %log_f3248.i to <8 x i32>
  %log_f3259.i = or <8 x i32> %log_f3257.i, %log_f321.i
  %log_f3260.i = bitcast <8 x i32> %log_f3259.i to <8 x float>
  %log_f3263.i = or <8 x i32> %log_f324.i, %log_f327.i
  %log_f3264.i = bitcast <8 x i32> %log_f3263.i to <8 x float>
  %log_f3266.i = xor <8 x i32> %log_f3263.i, splat (i32 -1)
  %61 = bitcast <8 x i32> %log_f3266.i to <8 x float>
  %log_f3269.i = and <8 x i32> %log_f3266.i, %log_f3259.i
  %62 = bitcast <8 x i32> %log_f3269.i to <8 x float>
  %log_f3272.i = or <8 x i32> %log_f3255.i, %log_f3269.i
  %log_f3273.i = bitcast <8 x i32> %log_f3272.i to <8 x float>
  %63 = fmul <8 x float> %45, %45
  %64 = fmul <8 x float> %45, zeroinitializer
  %65 = fadd <8 x float> %64, splat (float 1.000000e+00)
  %66 = fmul <8 x float> %45, %65
  %67 = fadd <8 x float> %66, splat (float 0x402E2035A0000000)
  %68 = fmul <8 x float> %45, %67
  %69 = fadd <8 x float> %68, splat (float 0x4054C30B60000000)
  %70 = fmul <8 x float> %45, %69
  %71 = fadd <8 x float> %70, splat (float 0x406BB865A0000000)
  %72 = fmul <8 x float> %45, %71
  %73 = fadd <8 x float> %72, splat (float 0x4073519460000000)
  %74 = fmul <8 x float> %45, %73
  %75 = fadd <8 x float> %74, splat (float 0x406B0DB140000000)
  %76 = fmul <8 x float> %45, %75
  %77 = fadd <8 x float> %76, splat (float 0x404E0F3040000000)
  %78 = fadd <8 x float> %64, splat (float 0x3F07BC0960000000)
  %79 = fmul <8 x float> %45, %78
  %80 = fadd <8 x float> %79, splat (float 0x3FDFE818A0000000)
  %81 = fmul <8 x float> %45, %80
  %82 = fadd <8 x float> %81, splat (float 0x401A509F40000000)
  %83 = fmul <8 x float> %45, %82
  %84 = fadd <8 x float> %83, splat (float 0x403DE97380000000)
  %85 = fmul <8 x float> %45, %84
  %86 = fadd <8 x float> %85, splat (float 0x404E798EC0000000)
  %87 = fmul <8 x float> %45, %86
  %88 = fadd <8 x float> %87, splat (float 0x404C8E75A0000000)
  %89 = fmul <8 x float> %45, %88
  %90 = fadd <8 x float> %89, splat (float 0x40340A2020000000)
  %91 = fdiv <8 x float> %90, %77
  %92 = fmul <8 x float> %45, %63
  %93 = fmul <8 x float> %92, %91
  %94 = fmul <8 x float> %63, splat (float 5.000000e-01)
  %95 = fsub <8 x float> %93, %94
  %96 = fadd <8 x float> %45, %95
  %97 = tail call <8 x float> @llvm.fabs.v8f32(<8 x float> %45)
  %98 = fcmp olt <8 x float> %97, splat (float 0x3FDA8279A0000000)
  %99 = select <8 x i1> %98, <8 x float> %96, <8 x float> %log_f3273.i
  %100 = fneg <8 x float> %99
  %101 = fcmp ogt <8 x float> %99, splat (float -5.000000e+00)
  %102 = select <8 x i1> %101, <8 x float> splat (float 0x3FF805C5E0000000), <8 x float> splat (float 0x4006A9EFC0000000)
  %103 = select <8 x i1> %101, <8 x float> splat (float 0x3FCF91EC60000000), <8 x float> splat (float 0x3FF006DB60000000)
  %104 = select <8 x i1> %101, <8 x float> splat (float 0xBF711C9DE0000000), <8 x float> splat (float 0x3F8354AFC0000000)
  %105 = select <8 x i1> %101, <8 x float> splat (float 0xBF548A8100000000), <8 x float> splat (float 0xBF7F38BAE0000000)
  %106 = select <8 x i1> %101, <8 x float> splat (float 0x3F2CA65B60000000), <8 x float> splat (float 0x3F77824F60000000)
  %107 = select <8 x i1> %101, <8 x float> splat (float 0xBED26B5820000000), <8 x float> splat (float 0xBF6E17BCE0000000)
  %108 = select <8 x i1> %101, <8 x float> splat (float 0xBECD8E6AE0000000), <8 x float> splat (float 0x3F561B8E40000000)
  %109 = select <8 x i1> %101, <8 x float> splat (float 0x3E970966C0000000), <8 x float> splat (float 0x3F1A76AD60000000)
  %110 = select <8 x i1> %101, <8 x float> splat (float 0x3E5E2CB100000000), <8 x float> splat (float 0xBF2A3E1360000000)
  %111 = fsub <8 x float> splat (float -2.500000e+00), %99
  %112 = tail call <8 x float> @llvm.sqrt.v8f32(<8 x float> %100)
  %113 = fadd <8 x float> %112, splat (float -3.000000e+00)
  %114 = select <8 x i1> %101, <8 x float> %111, <8 x float> %113
  %115 = fmul <8 x float> %110, %114
  %116 = fadd <8 x float> %109, %115
  %117 = fmul <8 x float> %114, %116
  %118 = fadd <8 x float> %108, %117
  %119 = fmul <8 x float> %114, %118
  %120 = fadd <8 x float> %107, %119
  %121 = fmul <8 x float> %114, %120
  %122 = fadd <8 x float> %106, %121
  %123 = fmul <8 x float> %114, %122
  %124 = fadd <8 x float> %105, %123
  %125 = fmul <8 x float> %114, %124
  %126 = fadd <8 x float> %104, %125
  %127 = fmul <8 x float> %114, %126
  %128 = fadd <8 x float> %103, %127
  %129 = fmul <8 x float> %114, %128
  %130 = fadd <8 x float> %102, %129
  %131 = select <8 x i1> %43, <8 x float> splat (float 0x7FF0000000000000), <8 x float> %130
  %132 = fmul <8 x float> %41, %131
  %133 = bitcast <8 x float> %132 to <8 x i32>
  %134 = lshr <8 x i32> %133, splat (i32 16)
  %135 = and <8 x i32> %134, splat (i32 1)
  %136 = fcmp uno <8 x float> %132, zeroinitializer
  %137 = and <8 x i32> %133, splat (i32 -8388608)
  %138 = or disjoint <8 x i32> %137, splat (i32 4194304)
  %139 = add <8 x i32> %133, splat (i32 32767)
  %140 = add <8 x i32> %139, %135
  %141 = and <8 x i32> %140, splat (i32 -65536)
  %142 = select <8 x i1> %136, <8 x i32> %138, <8 x i32> %141
  %143 = bitcast <8 x i32> %142 to <8 x float>
  %144 = fmul <8 x float> %143, splat (float 0x3FF6A00000000000)
  %145 = bitcast <8 x float> %144 to <8 x i32>
  %146 = lshr <8 x i32> %145, splat (i32 16)
  %147 = and <8 x i32> %146, splat (i32 1)
  %148 = fcmp uno <8 x float> %144, zeroinitializer
  %149 = and <8 x i32> %145, splat (i32 -8388608)
  %150 = or disjoint <8 x i32> %149, splat (i32 4194304)
  %151 = add <8 x i32> %145, splat (i32 32767)
  %152 = add <8 x i32> %151, %147
  %153 = select <8 x i1> %148, <8 x i32> %150, <8 x i32> %152
  %154 = lshr <8 x i32> %153, splat (i32 16)
  %155 = trunc nuw <8 x i32> %154 to <8 x i16>
  %156 = getelementptr inbounds [512 x [512 x bfloat]], ptr %arg2, i64 0, i64 %multiply_convert_fusion.clone.invar_address.dim.0.03, i64 %index
  store <8 x i16> %155, ptr %156, align 16, !alias.scope !5
  %index.next = add nuw i64 %index, 8
  %157 = icmp eq i64 %index.next, 512
  br i1 %157, label %multiply_convert_fusion.clone.loop_exit.dim.1, label %vector.body, !llvm.loop !8

multiply_convert_fusion.clone.loop_exit.dim.1:    ; preds = %vector.body
  %invar.inc = add nuw nsw i64 %multiply_convert_fusion.clone.invar_address.dim.0.03, 1
  %exitcond4.not = icmp eq i64 %invar.inc, %up_dim_0
  br i1 %exitcond4.not, label %return, label %vector.ph, !llvm.loop !11

return:                                           ; preds = %multiply_convert_fusion.clone.loop_exit.dim.1, %1
  ret ptr null
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <8 x float> @llvm.maximum.v8f32(<8 x float>, <8 x float>) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <8 x float> @llvm.fabs.v8f32(<8 x float>) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <8 x float> @llvm.sqrt.v8f32(<8 x float>) #1

attributes #0 = { nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable "frame-pointer"="all" "prefer-vector-width"="256" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"xla_dylib_index", i64 5}
!1 = !{}
!2 = !{i64 1048576}
!3 = !{i64 64}
!4 = !{i64 524288}
!5 = !{!6}
!6 = !{!"result slice: {index:0, offset:0, size:524288}", !7}
!7 = !{!"XLA host kernel multiply_convert_fusion.clone AA domain"}
!8 = distinct !{!8, !9, !10}
!9 = !{!"llvm.loop.isvectorized", i32 1}
!10 = !{!"llvm.loop.unroll.runtime.disable"}
!11 = distinct !{!11, !12}
!12 = !{!"llvm.loop.unroll.disable"}

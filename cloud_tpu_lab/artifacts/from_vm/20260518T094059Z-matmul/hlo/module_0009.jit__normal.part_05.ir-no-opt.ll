; ModuleID = '__compute_module_part_05'
source_filename = "__compute_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%XLA_CPU_KernelCallFrame = type { ptr, ptr, i64, ptr }
%XLA_CPU_NumWorkGroups = type { i64, i64, i64 }
%XLA_CPU_WorkGroupId = type { i64, i64, i64 }
%XLA_CPU_KernelArg = type { ptr, i64 }

@__llvmsplit_unnamed.12 = private unnamed_addr constant [4 x i8] c"\00\00\B5?"
@__llvmsplit_unnamed.13 = private unnamed_addr constant [4 x i8] c"\00\00\7F\BF"
@__llvmsplit_unnamed.14 = private unnamed_addr constant [4 x i8] c"\00\00\00@"
@__llvmsplit_unnamed.15 = private unnamed_addr constant [4 x i8] c"\00\00\80\BF"
@__llvmsplit_unnamed.16 = private unnamed_addr constant [2 x i8] c"\80?"
@__llvmsplit_unnamed.17 = private unnamed_addr constant [2 x i8] c"\01\00"
@__llvmsplit_unnamed.18 = private unnamed_addr constant [4 x i8] c"\00\00@\C0"
@__llvmsplit_unnamed.19 = private unnamed_addr constant [4 x i8] c"\00\00 \C0"
@__llvmsplit_unnamed.20 = private unnamed_addr constant [4 x i8] c"\00\00\A0@"
@__llvmsplit_unnamed.21 = private unnamed_addr constant [4 x i8] c"\9B\F0Q\B9"
@__llvmsplit_unnamed.22 = private unnamed_addr constant [4 x i8] c"\88e\F12"
@__llvmsplit_unnamed.23 = private unnamed_addr constant [4 x i8] c"k\B5\D38"
@__llvmsplit_unnamed.24 = private unnamed_addr constant [4 x i8] c"6K\B84"
@__llvmsplit_unnamed.25 = private unnamed_addr constant [4 x i8] c"r\DC\B0:"
@__llvmsplit_unnamed.26 = private unnamed_addr constant [4 x i8] c"Wsl\B6"
@__llvmsplit_unnamed.27 = private unnamed_addr constant [4 x i8] c"\E7\BDp\BB"
@__llvmsplit_unnamed.28 = private unnamed_addr constant [4 x i8] c"\C1Z\93\B6"
@__llvmsplit_unnamed.29 = private unnamed_addr constant [4 x i8] c"{\12\BC;"
@__llvmsplit_unnamed.30 = private unnamed_addr constant [4 x i8] c"\DB2e9"
@__llvmsplit_unnamed.31 = private unnamed_addr constant [4 x i8] c"\D7\C5\F9\BB"
@__llvmsplit_unnamed.32 = private unnamed_addr constant [4 x i8] c"\08T\A4\BA"
@__llvmsplit_unnamed.33 = private unnamed_addr constant [4 x i8] c"~\A5\1A<"
@__llvmsplit_unnamed.34 = private unnamed_addr constant [4 x i8] c"\EF\E4\88\BB"
@__llvmsplit_unnamed.35 = private unnamed_addr constant [4 x i8] c"\DB6\80?"
@__llvmsplit_unnamed.36 = private unnamed_addr constant [4 x i8] c"c\8F|>"
@__llvmsplit_unnamed.37 = private unnamed_addr constant [4 x i8] c"~O5@"
@__llvmsplit_unnamed.38 = private unnamed_addr constant [4 x i8] c"/.\C0?"
@__llvmsplit_unnamed.39 = private unnamed_addr constant [4 x i8] c"\00\00\80\7F"
@__llvmsplit_unnamed.40 = private unnamed_addr constant [4 x i8] c"\00\00\80?"
@multiply_convert_fusion.clone_parallel_bounds = private constant [24 x [1 x [2 x i64]]] [[1 x [2 x i64]] [[2 x i64] [i64 0, i64 21]], [1 x [2 x i64]] [[2 x i64] [i64 21, i64 42]], [1 x [2 x i64]] [[2 x i64] [i64 42, i64 63]], [1 x [2 x i64]] [[2 x i64] [i64 63, i64 84]], [1 x [2 x i64]] [[2 x i64] [i64 84, i64 105]], [1 x [2 x i64]] [[2 x i64] [i64 105, i64 126]], [1 x [2 x i64]] [[2 x i64] [i64 126, i64 147]], [1 x [2 x i64]] [[2 x i64] [i64 147, i64 168]], [1 x [2 x i64]] [[2 x i64] [i64 168, i64 189]], [1 x [2 x i64]] [[2 x i64] [i64 189, i64 210]], [1 x [2 x i64]] [[2 x i64] [i64 210, i64 231]], [1 x [2 x i64]] [[2 x i64] [i64 231, i64 252]], [1 x [2 x i64]] [[2 x i64] [i64 252, i64 273]], [1 x [2 x i64]] [[2 x i64] [i64 273, i64 294]], [1 x [2 x i64]] [[2 x i64] [i64 294, i64 315]], [1 x [2 x i64]] [[2 x i64] [i64 315, i64 336]], [1 x [2 x i64]] [[2 x i64] [i64 336, i64 357]], [1 x [2 x i64]] [[2 x i64] [i64 357, i64 378]], [1 x [2 x i64]] [[2 x i64] [i64 378, i64 399]], [1 x [2 x i64]] [[2 x i64] [i64 399, i64 420]], [1 x [2 x i64]] [[2 x i64] [i64 420, i64 441]], [1 x [2 x i64]] [[2 x i64] [i64 441, i64 462]], [1 x [2 x i64]] [[2 x i64] [i64 462, i64 483]], [1 x [2 x i64]] [[2 x i64] [i64 483, i64 512]]]

; Function Attrs: uwtable
define ptr @multiply_convert_fusion.clone(ptr %0) #0 {
  %multiply_convert_fusion.clone.invar_address.dim.1 = alloca i64, align 8
  %multiply_convert_fusion.clone.invar_address.dim.0 = alloca i64, align 8
  %num_workgroups_gep = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 0
  %num_workgroups = load ptr, ptr %num_workgroups_gep, align 8
  %num_workgroups_x_gep = getelementptr inbounds nuw %XLA_CPU_NumWorkGroups, ptr %num_workgroups, i32 0, i32 0
  %num_workgroups_y_gep = getelementptr inbounds nuw %XLA_CPU_NumWorkGroups, ptr %num_workgroups, i32 0, i32 1
  %num_workgroups_z_gep = getelementptr inbounds nuw %XLA_CPU_NumWorkGroups, ptr %num_workgroups, i32 0, i32 2
  %num_workgroups_x = load i64, ptr %num_workgroups_x_gep, align 4
  %num_workgroups_y = load i64, ptr %num_workgroups_y_gep, align 4
  %num_workgroups_z = load i64, ptr %num_workgroups_z_gep, align 4
  %workgroup_id_gep = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 1
  %workgroup_id = load ptr, ptr %workgroup_id_gep, align 8
  %workgroup_id_x_gep = getelementptr inbounds nuw %XLA_CPU_WorkGroupId, ptr %workgroup_id, i32 0, i32 0
  %workgroup_id_y_gep = getelementptr inbounds nuw %XLA_CPU_WorkGroupId, ptr %workgroup_id, i32 0, i32 1
  %workgroup_id_z_gep = getelementptr inbounds nuw %XLA_CPU_WorkGroupId, ptr %workgroup_id, i32 0, i32 2
  %workgroup_id_x = load i64, ptr %workgroup_id_x_gep, align 4
  %workgroup_id_y = load i64, ptr %workgroup_id_y_gep, align 4
  %workgroup_id_z = load i64, ptr %workgroup_id_z_gep, align 4
  %args_gep = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args = load ptr, ptr %args_gep, align 8
  %arg0_gep = getelementptr %XLA_CPU_KernelArg, ptr %args, i32 0, i32 0
  %arg0 = load ptr, ptr %arg0_gep, align 8, !invariant.load !1, !dereferenceable !2, !align !3
  %args_gep1 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args2 = load ptr, ptr %args_gep1, align 8
  %arg1_gep = getelementptr %XLA_CPU_KernelArg, ptr %args2, i32 1, i32 0
  %arg1 = load ptr, ptr %arg1_gep, align 8, !invariant.load !1, !dereferenceable !2, !align !3
  %args_gep3 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args4 = load ptr, ptr %args_gep3, align 8
  %arg2_gep = getelementptr %XLA_CPU_KernelArg, ptr %args4, i32 2, i32 0
  %arg2 = load ptr, ptr %arg2_gep, align 8, !invariant.load !1, !dereferenceable !4, !align !3
  %lo_dim_0_gep = getelementptr inbounds [24 x [1 x [2 x i64]]], ptr @multiply_convert_fusion.clone_parallel_bounds, i32 0, i64 %workgroup_id_x, i32 0, i32 0
  %up_dim_0_gep = getelementptr inbounds [24 x [1 x [2 x i64]]], ptr @multiply_convert_fusion.clone_parallel_bounds, i32 0, i64 %workgroup_id_x, i32 0, i32 1
  %lo_dim_0 = load i64, ptr %lo_dim_0_gep, align 4
  %up_dim_0 = load i64, ptr %up_dim_0_gep, align 4
  store i64 %lo_dim_0, ptr %multiply_convert_fusion.clone.invar_address.dim.0, align 4
  br label %multiply_convert_fusion.clone.loop_header.dim.0

multiply_convert_fusion.clone.loop_header.dim.0:  ; preds = %multiply_convert_fusion.clone.loop_exit.dim.1, %1
  %multiply_convert_fusion.clone.indvar.dim.0 = load i64, ptr %multiply_convert_fusion.clone.invar_address.dim.0, align 4
  %2 = icmp uge i64 %multiply_convert_fusion.clone.indvar.dim.0, %up_dim_0
  br i1 %2, label %multiply_convert_fusion.clone.loop_exit.dim.0, label %multiply_convert_fusion.clone.loop_body.dim.0

multiply_convert_fusion.clone.loop_body.dim.0:    ; preds = %multiply_convert_fusion.clone.loop_header.dim.0
  store i64 0, ptr %multiply_convert_fusion.clone.invar_address.dim.1, align 4
  br label %multiply_convert_fusion.clone.loop_header.dim.1

multiply_convert_fusion.clone.loop_header.dim.1:  ; preds = %multiply_convert_fusion.clone.loop_body.dim.1, %multiply_convert_fusion.clone.loop_body.dim.0
  %multiply_convert_fusion.clone.indvar.dim.1 = load i64, ptr %multiply_convert_fusion.clone.invar_address.dim.1, align 4
  %3 = icmp uge i64 %multiply_convert_fusion.clone.indvar.dim.1, 512
  br i1 %3, label %multiply_convert_fusion.clone.loop_exit.dim.1, label %multiply_convert_fusion.clone.loop_body.dim.1

multiply_convert_fusion.clone.loop_body.dim.1:    ; preds = %multiply_convert_fusion.clone.loop_header.dim.1
  %constant.191 = load float, ptr @__llvmsplit_unnamed.13, align 4
  %4 = getelementptr inbounds [512 x [512 x i32]], ptr %arg0, i64 0, i64 %multiply_convert_fusion.clone.indvar.dim.0, i64 %multiply_convert_fusion.clone.indvar.dim.1
  %5 = load i32, ptr %4, align 4, !invariant.load !1, !noalias !5
  %6 = getelementptr inbounds [512 x [512 x i32]], ptr %arg1, i64 0, i64 %multiply_convert_fusion.clone.indvar.dim.0, i64 %multiply_convert_fusion.clone.indvar.dim.1
  %7 = load i32, ptr %6, align 4, !invariant.load !1, !noalias !5
  %8 = xor i32 %5, %7
  %9 = trunc i32 %8 to i8
  %10 = zext i8 %9 to i16
  %constant.190 = load i16, ptr @__llvmsplit_unnamed.17, align 2
  %11 = lshr i16 %10, %constant.190
  %shft.chk = icmp ult i16 %constant.190, 16
  %12 = select i1 %shft.chk, i16 %11, i16 0
  %constant.189 = load i16, ptr @__llvmsplit_unnamed.16, align 2
  %13 = or i16 %12, %constant.189
  %14 = bitcast i16 %13 to bfloat
  %15 = bitcast bfloat %14 to i16
  %16 = zext i16 %15 to i32
  %17 = shl i32 %16, 16
  %18 = bitcast i32 %17 to float
  %constant.188 = load float, ptr @__llvmsplit_unnamed.15, align 4
  %add.117 = fadd float %18, %constant.188
  %19 = bitcast float %add.117 to i32
  %20 = lshr i32 %19, 16
  %21 = and i32 %20, 1
  %22 = add i32 32767, %21
  %23 = call i1 @llvm.is.fpclass.f32(float %add.117, i32 3)
  %24 = and i32 %19, -4194304
  %25 = or i32 %24, 4194304
  %26 = add i32 %19, %22
  %27 = select i1 %23, i32 %25, i32 %26
  %28 = lshr i32 %27, 16
  %29 = trunc i32 %28 to i16
  %30 = bitcast i16 %29 to bfloat
  %31 = bitcast bfloat %30 to i16
  %32 = zext i16 %31 to i32
  %33 = shl i32 %32, 16
  %34 = bitcast i32 %33 to float
  %constant.187 = load float, ptr @__llvmsplit_unnamed.14, align 4
  %multiply.48 = fmul float %34, %constant.187
  %35 = bitcast float %multiply.48 to i32
  %36 = lshr i32 %35, 16
  %37 = and i32 %36, 1
  %38 = add i32 32767, %37
  %39 = call i1 @llvm.is.fpclass.f32(float %multiply.48, i32 3)
  %40 = and i32 %35, -4194304
  %41 = or i32 %40, 4194304
  %42 = add i32 %35, %38
  %43 = select i1 %39, i32 %41, i32 %42
  %44 = lshr i32 %43, 16
  %45 = trunc i32 %44 to i16
  %46 = bitcast i16 %45 to bfloat
  %47 = bitcast bfloat %46 to i16
  %48 = zext i16 %47 to i32
  %49 = shl i32 %48, 16
  %50 = bitcast i32 %49 to float
  %add.115 = fadd float %50, %constant.191
  %51 = bitcast float %add.115 to i32
  %52 = lshr i32 %51, 16
  %53 = and i32 %52, 1
  %54 = add i32 32767, %53
  %55 = call i1 @llvm.is.fpclass.f32(float %add.115, i32 3)
  %56 = and i32 %51, -4194304
  %57 = or i32 %56, 4194304
  %58 = add i32 %51, %54
  %59 = select i1 %55, i32 %57, i32 %58
  %60 = lshr i32 %59, 16
  %61 = trunc i32 %60 to i16
  %62 = bitcast i16 %61 to bfloat
  %63 = bitcast bfloat %62 to i16
  %64 = zext i16 %63 to i32
  %65 = shl i32 %64, 16
  %66 = bitcast i32 %65 to float
  %67 = call float @llvm.maximum.f32(float %constant.191, float %66)
  %68 = call float @llvm.fabs.f32(float %67)
  %constant.185 = load float, ptr @__llvmsplit_unnamed.40, align 4
  %compare.7 = fcmp oeq float %68, %constant.185
  %69 = zext i1 %compare.7 to i8
  %constant.184 = load float, ptr @__llvmsplit_unnamed.39, align 4
  %multiply.47 = fmul float %67, %constant.184
  %70 = fneg float %67
  %multiply.46 = fmul float %67, %70
  %71 = fadd float %multiply.46, 1.000000e+00
  %72 = call float @llvm.log.f32(float %71)
  %73 = fmul float %multiply.46, %multiply.46
  %74 = fmul float 0.000000e+00, %multiply.46
  %75 = fadd float %74, 1.000000e+00
  %76 = fmul float %75, %multiply.46
  %77 = fadd float %76, 0x402E2035A0000000
  %78 = fmul float %77, %multiply.46
  %79 = fadd float %78, 0x4054C30B60000000
  %80 = fmul float %79, %multiply.46
  %81 = fadd float %80, 0x406BB865A0000000
  %82 = fmul float %81, %multiply.46
  %83 = fadd float %82, 0x4073519460000000
  %84 = fmul float %83, %multiply.46
  %85 = fadd float %84, 0x406B0DB140000000
  %86 = fmul float %85, %multiply.46
  %87 = fadd float %86, 0x404E0F3040000000
  %88 = fmul float 0.000000e+00, %multiply.46
  %89 = fadd float %88, 0x3F07BC0960000000
  %90 = fmul float %89, %multiply.46
  %91 = fadd float %90, 0x3FDFE818A0000000
  %92 = fmul float %91, %multiply.46
  %93 = fadd float %92, 0x401A509F40000000
  %94 = fmul float %93, %multiply.46
  %95 = fadd float %94, 0x403DE97380000000
  %96 = fmul float %95, %multiply.46
  %97 = fadd float %96, 0x404E798EC0000000
  %98 = fmul float %97, %multiply.46
  %99 = fadd float %98, 0x404C8E75A0000000
  %100 = fmul float %99, %multiply.46
  %101 = fadd float %100, 0x40340A2020000000
  %102 = fdiv float %101, %87
  %103 = fmul float %multiply.46, %73
  %104 = fmul float %103, %102
  %105 = fmul float -5.000000e-01, %73
  %106 = fadd float %105, %104
  %107 = fadd float %multiply.46, %106
  %108 = call float @llvm.fabs.f32(float %multiply.46)
  %109 = fcmp olt float %108, 0x3FDA8279A0000000
  %110 = select i1 %109, float %107, float %72
  %111 = fneg float %110
  %constant.183 = load float, ptr @__llvmsplit_unnamed.20, align 4
  %compare.6 = fcmp olt float %111, %constant.183
  %112 = zext i1 %compare.6 to i8
  %constant.182 = load float, ptr @__llvmsplit_unnamed.38, align 4
  %constant.181 = load float, ptr @__llvmsplit_unnamed.37, align 4
  %113 = trunc i8 %112 to i1
  %114 = select i1 %113, float %constant.182, float %constant.181
  %constant.180 = load float, ptr @__llvmsplit_unnamed.36, align 4
  %constant.179 = load float, ptr @__llvmsplit_unnamed.35, align 4
  %115 = trunc i8 %112 to i1
  %116 = select i1 %115, float %constant.180, float %constant.179
  %constant.178 = load float, ptr @__llvmsplit_unnamed.34, align 4
  %constant.177 = load float, ptr @__llvmsplit_unnamed.33, align 4
  %117 = trunc i8 %112 to i1
  %118 = select i1 %117, float %constant.178, float %constant.177
  %constant.176 = load float, ptr @__llvmsplit_unnamed.32, align 4
  %constant.175 = load float, ptr @__llvmsplit_unnamed.31, align 4
  %119 = trunc i8 %112 to i1
  %120 = select i1 %119, float %constant.176, float %constant.175
  %constant.174 = load float, ptr @__llvmsplit_unnamed.30, align 4
  %constant.173 = load float, ptr @__llvmsplit_unnamed.29, align 4
  %121 = trunc i8 %112 to i1
  %122 = select i1 %121, float %constant.174, float %constant.173
  %constant.172 = load float, ptr @__llvmsplit_unnamed.28, align 4
  %constant.171 = load float, ptr @__llvmsplit_unnamed.27, align 4
  %123 = trunc i8 %112 to i1
  %124 = select i1 %123, float %constant.172, float %constant.171
  %constant.170 = load float, ptr @__llvmsplit_unnamed.26, align 4
  %constant.169 = load float, ptr @__llvmsplit_unnamed.25, align 4
  %125 = trunc i8 %112 to i1
  %126 = select i1 %125, float %constant.170, float %constant.169
  %constant.168 = load float, ptr @__llvmsplit_unnamed.24, align 4
  %constant.167 = load float, ptr @__llvmsplit_unnamed.23, align 4
  %127 = trunc i8 %112 to i1
  %128 = select i1 %127, float %constant.168, float %constant.167
  %constant.166 = load float, ptr @__llvmsplit_unnamed.22, align 4
  %constant.165 = load float, ptr @__llvmsplit_unnamed.21, align 4
  %129 = trunc i8 %112 to i1
  %130 = select i1 %129, float %constant.166, float %constant.165
  %constant.164 = load float, ptr @__llvmsplit_unnamed.19, align 4
  %add.114 = fadd float %111, %constant.164
  %131 = call float @llvm.sqrt.f32(float %111)
  %constant.163 = load float, ptr @__llvmsplit_unnamed.18, align 4
  %add.112 = fadd float %131, %constant.163
  %132 = trunc i8 %112 to i1
  %133 = select i1 %132, float %add.114, float %add.112
  %multiply.45 = fmul float %130, %133
  %add.111 = fadd float %128, %multiply.45
  %multiply.44 = fmul float %add.111, %133
  %add.110 = fadd float %126, %multiply.44
  %multiply.43 = fmul float %add.110, %133
  %add.109 = fadd float %124, %multiply.43
  %multiply.42 = fmul float %add.109, %133
  %add.108 = fadd float %122, %multiply.42
  %multiply.41 = fmul float %add.108, %133
  %add.107 = fadd float %120, %multiply.41
  %multiply.40 = fmul float %add.107, %133
  %add.106 = fadd float %118, %multiply.40
  %multiply.39 = fmul float %add.106, %133
  %add.105 = fadd float %116, %multiply.39
  %multiply.38 = fmul float %add.105, %133
  %add.104 = fadd float %114, %multiply.38
  %multiply.37 = fmul float %add.104, %67
  %134 = trunc i8 %69 to i1
  %135 = select i1 %134, float %multiply.47, float %multiply.37
  %136 = bitcast float %135 to i32
  %137 = lshr i32 %136, 16
  %138 = and i32 %137, 1
  %139 = add i32 32767, %138
  %140 = call i1 @llvm.is.fpclass.f32(float %135, i32 3)
  %141 = and i32 %136, -4194304
  %142 = or i32 %141, 4194304
  %143 = add i32 %136, %139
  %144 = select i1 %140, i32 %142, i32 %143
  %145 = lshr i32 %144, 16
  %146 = trunc i32 %145 to i16
  %147 = bitcast i16 %146 to bfloat
  %148 = bitcast bfloat %147 to i16
  %149 = zext i16 %148 to i32
  %150 = shl i32 %149, 16
  %151 = bitcast i32 %150 to float
  %constant.162 = load float, ptr @__llvmsplit_unnamed.12, align 4
  %multiply.36 = fmul float %151, %constant.162
  %152 = bitcast float %multiply.36 to i32
  %153 = lshr i32 %152, 16
  %154 = and i32 %153, 1
  %155 = add i32 32767, %154
  %156 = call i1 @llvm.is.fpclass.f32(float %multiply.36, i32 3)
  %157 = and i32 %152, -4194304
  %158 = or i32 %157, 4194304
  %159 = add i32 %152, %155
  %160 = select i1 %156, i32 %158, i32 %159
  %161 = lshr i32 %160, 16
  %162 = trunc i32 %161 to i16
  %163 = bitcast i16 %162 to bfloat
  %164 = getelementptr inbounds [512 x [512 x bfloat]], ptr %arg2, i64 0, i64 %multiply_convert_fusion.clone.indvar.dim.0, i64 %multiply_convert_fusion.clone.indvar.dim.1
  store bfloat %163, ptr %164, align 2, !alias.scope !5
  %invar.inc5 = add nuw nsw i64 %multiply_convert_fusion.clone.indvar.dim.1, 1
  store i64 %invar.inc5, ptr %multiply_convert_fusion.clone.invar_address.dim.1, align 4
  br label %multiply_convert_fusion.clone.loop_header.dim.1

multiply_convert_fusion.clone.loop_exit.dim.1:    ; preds = %multiply_convert_fusion.clone.loop_header.dim.1
  %invar.inc = add nuw nsw i64 %multiply_convert_fusion.clone.indvar.dim.0, 1
  store i64 %invar.inc, ptr %multiply_convert_fusion.clone.invar_address.dim.0, align 4
  br label %multiply_convert_fusion.clone.loop_header.dim.0, !llvm.loop !8

multiply_convert_fusion.clone.loop_exit.dim.0:    ; preds = %multiply_convert_fusion.clone.loop_header.dim.0
  br label %return

return:                                           ; preds = %multiply_convert_fusion.clone.loop_exit.dim.0
  ret ptr null
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i1 @llvm.is.fpclass.f32(float, i32 immarg) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maximum.f32(float, float) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fabs.f32(float) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.log.f32(float) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.sqrt.f32(float) #1

attributes #0 = { uwtable "frame-pointer"="all" "prefer-vector-width"="256" }
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
!8 = distinct !{!8, !9}
!9 = !{!"llvm.loop.unroll.disable"}

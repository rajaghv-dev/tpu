; ModuleID = '__compute_module_part_04'
source_filename = "__compute_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@broadcast_add_fusion.clone_parallel_bounds = private unnamed_addr constant [24 x [1 x [2 x i64]]] [[1 x [2 x i64]] [[2 x i64] [i64 0, i64 21]], [1 x [2 x i64]] [[2 x i64] [i64 21, i64 42]], [1 x [2 x i64]] [[2 x i64] [i64 42, i64 63]], [1 x [2 x i64]] [[2 x i64] [i64 63, i64 84]], [1 x [2 x i64]] [[2 x i64] [i64 84, i64 105]], [1 x [2 x i64]] [[2 x i64] [i64 105, i64 126]], [1 x [2 x i64]] [[2 x i64] [i64 126, i64 147]], [1 x [2 x i64]] [[2 x i64] [i64 147, i64 168]], [1 x [2 x i64]] [[2 x i64] [i64 168, i64 189]], [1 x [2 x i64]] [[2 x i64] [i64 189, i64 210]], [1 x [2 x i64]] [[2 x i64] [i64 210, i64 231]], [1 x [2 x i64]] [[2 x i64] [i64 231, i64 252]], [1 x [2 x i64]] [[2 x i64] [i64 252, i64 273]], [1 x [2 x i64]] [[2 x i64] [i64 273, i64 294]], [1 x [2 x i64]] [[2 x i64] [i64 294, i64 315]], [1 x [2 x i64]] [[2 x i64] [i64 315, i64 336]], [1 x [2 x i64]] [[2 x i64] [i64 336, i64 357]], [1 x [2 x i64]] [[2 x i64] [i64 357, i64 378]], [1 x [2 x i64]] [[2 x i64] [i64 378, i64 399]], [1 x [2 x i64]] [[2 x i64] [i64 399, i64 420]], [1 x [2 x i64]] [[2 x i64] [i64 420, i64 441]], [1 x [2 x i64]] [[2 x i64] [i64 441, i64 462]], [1 x [2 x i64]] [[2 x i64] [i64 462, i64 483]], [1 x [2 x i64]] [[2 x i64] [i64 483, i64 512]]]

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define noalias noundef ptr @broadcast_add_fusion.clone(ptr readonly captures(none) %0) local_unnamed_addr #0 {
  %workgroup_id_gep = getelementptr inbounds nuw i8, ptr %0, i64 8
  %workgroup_id = load ptr, ptr %workgroup_id_gep, align 8
  %workgroup_id_x = load i64, ptr %workgroup_id, align 4
  %args_gep = getelementptr inbounds nuw i8, ptr %0, i64 24
  %args = load ptr, ptr %args_gep, align 8
  %arg2_gep = getelementptr i8, ptr %args, i64 32
  %arg2 = load ptr, ptr %arg2_gep, align 8, !invariant.load !1, !dereferenceable !2, !align !3
  %arg3_gep = getelementptr i8, ptr %args, i64 48
  %arg3 = load ptr, ptr %arg3_gep, align 8, !invariant.load !1, !dereferenceable !2, !align !3
  %arg5_gep = getelementptr i8, ptr %args, i64 80
  %arg5 = load ptr, ptr %arg5_gep, align 8, !invariant.load !1, !dereferenceable !2, !align !3
  %lo_dim_0_gep = getelementptr inbounds [24 x [1 x [2 x i64]]], ptr @broadcast_add_fusion.clone_parallel_bounds, i64 0, i64 %workgroup_id_x, i64 0, i64 0
  %up_dim_0_gep = getelementptr inbounds [24 x [1 x [2 x i64]]], ptr @broadcast_add_fusion.clone_parallel_bounds, i64 0, i64 %workgroup_id_x, i64 0, i64 1
  %lo_dim_0 = load i64, ptr %lo_dim_0_gep, align 16
  %up_dim_0 = load i64, ptr %up_dim_0_gep, align 8
  %.not2 = icmp ult i64 %lo_dim_0, %up_dim_0
  br i1 %.not2, label %broadcast_add_fusion.clone.loop_header.dim.1.preheader.lr.ph, label %return

broadcast_add_fusion.clone.loop_header.dim.1.preheader.lr.ph: ; preds = %1
  %arg4_gep = getelementptr i8, ptr %args, i64 64
  %arg4 = load ptr, ptr %arg4_gep, align 8, !invariant.load !1, !dereferenceable !4, !align !3
  %arg1_gep = getelementptr i8, ptr %args, i64 16
  %arg1 = load ptr, ptr %arg1_gep, align 8, !invariant.load !1, !dereferenceable !5, !align !3
  %arg0 = load ptr, ptr %args, align 8, !invariant.load !1, !dereferenceable !4, !align !3
  %2 = load i32, ptr %arg1, align 64, !invariant.load !1, !noalias !6
  %.fr = freeze i32 %2
  %shft.chk = icmp ult i32 %.fr, 32
  %3 = sub i32 32, %.fr
  %shft.chk12 = icmp ult i32 %3, 32
  %4 = getelementptr inbounds nuw i8, ptr %arg1, i64 4
  %5 = load i32, ptr %4, align 4, !invariant.load !1, !noalias !6
  %.fr4 = freeze i32 %5
  %shft.chk13 = icmp ult i32 %.fr4, 32
  %6 = sub i32 32, %.fr4
  %shft.chk15 = icmp ult i32 %6, 32
  %7 = getelementptr inbounds nuw i8, ptr %arg1, i64 8
  %8 = load i32, ptr %7, align 8, !invariant.load !1, !noalias !6
  %.fr5 = freeze i32 %8
  %shft.chk16 = icmp ult i32 %.fr5, 32
  %9 = sub i32 32, %.fr5
  %shft.chk18 = icmp ult i32 %9, 32
  %10 = getelementptr inbounds nuw i8, ptr %arg1, i64 12
  %11 = load i32, ptr %10, align 4, !invariant.load !1, !noalias !6
  %.fr6 = freeze i32 %11
  %shft.chk19 = icmp ult i32 %.fr6, 32
  %12 = sub i32 32, %.fr6
  %shft.chk21 = icmp ult i32 %12, 32
  %13 = load i32, ptr %arg0, align 64, !invariant.load !1, !noalias !6
  %14 = load i32, ptr %arg4, align 64, !invariant.load !1, !noalias !6
  %15 = add i32 %13, 1
  %16 = add i32 %15, %14
  br i1 %shft.chk, label %broadcast_add_fusion.clone.loop_header.dim.1.preheader.lr.ph.split.us, label %vector.ph.preheader

vector.ph.preheader:                              ; preds = %broadcast_add_fusion.clone.loop_header.dim.1.preheader.lr.ph
  %broadcast.splatinsert = insertelement <8 x i32> poison, i32 %3, i64 0
  %broadcast.splat = shufflevector <8 x i32> %broadcast.splatinsert, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert48 = insertelement <8 x i32> poison, i32 %.fr4, i64 0
  %broadcast.splat49 = shufflevector <8 x i32> %broadcast.splatinsert48, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert50 = insertelement <8 x i32> poison, i32 %6, i64 0
  %broadcast.splat51 = shufflevector <8 x i32> %broadcast.splatinsert50, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert52 = insertelement <8 x i32> poison, i32 %.fr5, i64 0
  %broadcast.splat53 = shufflevector <8 x i32> %broadcast.splatinsert52, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert54 = insertelement <8 x i32> poison, i32 %9, i64 0
  %broadcast.splat55 = shufflevector <8 x i32> %broadcast.splatinsert54, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert56 = insertelement <8 x i32> poison, i32 %.fr6, i64 0
  %broadcast.splat57 = shufflevector <8 x i32> %broadcast.splatinsert56, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert58 = insertelement <8 x i32> poison, i32 %12, i64 0
  %broadcast.splat59 = shufflevector <8 x i32> %broadcast.splatinsert58, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert60 = insertelement <8 x i32> poison, i32 %16, i64 0
  %broadcast.splat61 = shufflevector <8 x i32> %broadcast.splatinsert60, <8 x i32> poison, <8 x i32> zeroinitializer
  br label %vector.ph

broadcast_add_fusion.clone.loop_header.dim.1.preheader.lr.ph.split.us: ; preds = %broadcast_add_fusion.clone.loop_header.dim.1.preheader.lr.ph
  br i1 %shft.chk12, label %broadcast_add_fusion.clone.loop_header.dim.1.preheader.lr.ph.split.us.split.us, label %vector.ph64.preheader

vector.ph64.preheader:                            ; preds = %broadcast_add_fusion.clone.loop_header.dim.1.preheader.lr.ph.split.us
  %broadcast.splatinsert65 = insertelement <8 x i32> poison, i32 %.fr, i64 0
  %broadcast.splat66 = shufflevector <8 x i32> %broadcast.splatinsert65, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert67 = insertelement <8 x i32> poison, i32 %.fr4, i64 0
  %broadcast.splat68 = shufflevector <8 x i32> %broadcast.splatinsert67, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert69 = insertelement <8 x i32> poison, i32 %6, i64 0
  %broadcast.splat70 = shufflevector <8 x i32> %broadcast.splatinsert69, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert71 = insertelement <8 x i32> poison, i32 %.fr5, i64 0
  %broadcast.splat72 = shufflevector <8 x i32> %broadcast.splatinsert71, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert73 = insertelement <8 x i32> poison, i32 %9, i64 0
  %broadcast.splat74 = shufflevector <8 x i32> %broadcast.splatinsert73, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert75 = insertelement <8 x i32> poison, i32 %.fr6, i64 0
  %broadcast.splat76 = shufflevector <8 x i32> %broadcast.splatinsert75, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert77 = insertelement <8 x i32> poison, i32 %12, i64 0
  %broadcast.splat78 = shufflevector <8 x i32> %broadcast.splatinsert77, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert79 = insertelement <8 x i32> poison, i32 %16, i64 0
  %broadcast.splat80 = shufflevector <8 x i32> %broadcast.splatinsert79, <8 x i32> poison, <8 x i32> zeroinitializer
  br label %vector.ph64

broadcast_add_fusion.clone.loop_header.dim.1.preheader.lr.ph.split.us.split.us: ; preds = %broadcast_add_fusion.clone.loop_header.dim.1.preheader.lr.ph.split.us
  br i1 %shft.chk13, label %broadcast_add_fusion.clone.loop_header.dim.1.preheader.lr.ph.split.us.split.us.split.us, label %vector.ph89.preheader

vector.ph89.preheader:                            ; preds = %broadcast_add_fusion.clone.loop_header.dim.1.preheader.lr.ph.split.us.split.us
  %broadcast.splatinsert90 = insertelement <8 x i32> poison, i32 %.fr, i64 0
  %broadcast.splat91 = shufflevector <8 x i32> %broadcast.splatinsert90, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert92 = insertelement <8 x i32> poison, i32 %3, i64 0
  %broadcast.splat93 = shufflevector <8 x i32> %broadcast.splatinsert92, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert94 = insertelement <8 x i32> poison, i32 %6, i64 0
  %broadcast.splat95 = shufflevector <8 x i32> %broadcast.splatinsert94, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert96 = insertelement <8 x i32> poison, i32 %.fr5, i64 0
  %broadcast.splat97 = shufflevector <8 x i32> %broadcast.splatinsert96, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert98 = insertelement <8 x i32> poison, i32 %9, i64 0
  %broadcast.splat99 = shufflevector <8 x i32> %broadcast.splatinsert98, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert100 = insertelement <8 x i32> poison, i32 %.fr6, i64 0
  %broadcast.splat101 = shufflevector <8 x i32> %broadcast.splatinsert100, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert102 = insertelement <8 x i32> poison, i32 %12, i64 0
  %broadcast.splat103 = shufflevector <8 x i32> %broadcast.splatinsert102, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert104 = insertelement <8 x i32> poison, i32 %16, i64 0
  %broadcast.splat105 = shufflevector <8 x i32> %broadcast.splatinsert104, <8 x i32> poison, <8 x i32> zeroinitializer
  br label %vector.ph89

broadcast_add_fusion.clone.loop_header.dim.1.preheader.lr.ph.split.us.split.us.split.us: ; preds = %broadcast_add_fusion.clone.loop_header.dim.1.preheader.lr.ph.split.us.split.us
  br i1 %shft.chk15, label %broadcast_add_fusion.clone.loop_header.dim.1.preheader.lr.ph.split.us.split.us.split.us.split.us, label %vector.ph114.preheader

vector.ph114.preheader:                           ; preds = %broadcast_add_fusion.clone.loop_header.dim.1.preheader.lr.ph.split.us.split.us.split.us
  %broadcast.splatinsert115 = insertelement <8 x i32> poison, i32 %.fr, i64 0
  %broadcast.splat116 = shufflevector <8 x i32> %broadcast.splatinsert115, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert117 = insertelement <8 x i32> poison, i32 %3, i64 0
  %broadcast.splat118 = shufflevector <8 x i32> %broadcast.splatinsert117, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert119 = insertelement <8 x i32> poison, i32 %.fr4, i64 0
  %broadcast.splat120 = shufflevector <8 x i32> %broadcast.splatinsert119, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert121 = insertelement <8 x i32> poison, i32 %.fr5, i64 0
  %broadcast.splat122 = shufflevector <8 x i32> %broadcast.splatinsert121, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert123 = insertelement <8 x i32> poison, i32 %9, i64 0
  %broadcast.splat124 = shufflevector <8 x i32> %broadcast.splatinsert123, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert125 = insertelement <8 x i32> poison, i32 %.fr6, i64 0
  %broadcast.splat126 = shufflevector <8 x i32> %broadcast.splatinsert125, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert127 = insertelement <8 x i32> poison, i32 %12, i64 0
  %broadcast.splat128 = shufflevector <8 x i32> %broadcast.splatinsert127, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert129 = insertelement <8 x i32> poison, i32 %16, i64 0
  %broadcast.splat130 = shufflevector <8 x i32> %broadcast.splatinsert129, <8 x i32> poison, <8 x i32> zeroinitializer
  br label %vector.ph114

broadcast_add_fusion.clone.loop_header.dim.1.preheader.lr.ph.split.us.split.us.split.us.split.us: ; preds = %broadcast_add_fusion.clone.loop_header.dim.1.preheader.lr.ph.split.us.split.us.split.us
  br i1 %shft.chk16, label %broadcast_add_fusion.clone.loop_header.dim.1.preheader.lr.ph.split.us.split.us.split.us.split.us.split.us, label %vector.ph139.preheader

vector.ph139.preheader:                           ; preds = %broadcast_add_fusion.clone.loop_header.dim.1.preheader.lr.ph.split.us.split.us.split.us.split.us
  %broadcast.splatinsert140 = insertelement <8 x i32> poison, i32 %.fr, i64 0
  %broadcast.splat141 = shufflevector <8 x i32> %broadcast.splatinsert140, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert142 = insertelement <8 x i32> poison, i32 %3, i64 0
  %broadcast.splat143 = shufflevector <8 x i32> %broadcast.splatinsert142, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert144 = insertelement <8 x i32> poison, i32 %.fr4, i64 0
  %broadcast.splat145 = shufflevector <8 x i32> %broadcast.splatinsert144, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert146 = insertelement <8 x i32> poison, i32 %6, i64 0
  %broadcast.splat147 = shufflevector <8 x i32> %broadcast.splatinsert146, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert148 = insertelement <8 x i32> poison, i32 %9, i64 0
  %broadcast.splat149 = shufflevector <8 x i32> %broadcast.splatinsert148, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert150 = insertelement <8 x i32> poison, i32 %.fr6, i64 0
  %broadcast.splat151 = shufflevector <8 x i32> %broadcast.splatinsert150, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert152 = insertelement <8 x i32> poison, i32 %12, i64 0
  %broadcast.splat153 = shufflevector <8 x i32> %broadcast.splatinsert152, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert154 = insertelement <8 x i32> poison, i32 %16, i64 0
  %broadcast.splat155 = shufflevector <8 x i32> %broadcast.splatinsert154, <8 x i32> poison, <8 x i32> zeroinitializer
  br label %vector.ph139

broadcast_add_fusion.clone.loop_header.dim.1.preheader.lr.ph.split.us.split.us.split.us.split.us.split.us: ; preds = %broadcast_add_fusion.clone.loop_header.dim.1.preheader.lr.ph.split.us.split.us.split.us.split.us
  br i1 %shft.chk18, label %broadcast_add_fusion.clone.loop_header.dim.1.preheader.lr.ph.split.us.split.us.split.us.split.us.split.us.split.us, label %vector.ph164.preheader

vector.ph164.preheader:                           ; preds = %broadcast_add_fusion.clone.loop_header.dim.1.preheader.lr.ph.split.us.split.us.split.us.split.us.split.us
  %broadcast.splatinsert165 = insertelement <8 x i32> poison, i32 %.fr, i64 0
  %broadcast.splat166 = shufflevector <8 x i32> %broadcast.splatinsert165, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert167 = insertelement <8 x i32> poison, i32 %3, i64 0
  %broadcast.splat168 = shufflevector <8 x i32> %broadcast.splatinsert167, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert169 = insertelement <8 x i32> poison, i32 %.fr4, i64 0
  %broadcast.splat170 = shufflevector <8 x i32> %broadcast.splatinsert169, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert171 = insertelement <8 x i32> poison, i32 %6, i64 0
  %broadcast.splat172 = shufflevector <8 x i32> %broadcast.splatinsert171, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert173 = insertelement <8 x i32> poison, i32 %.fr5, i64 0
  %broadcast.splat174 = shufflevector <8 x i32> %broadcast.splatinsert173, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert175 = insertelement <8 x i32> poison, i32 %.fr6, i64 0
  %broadcast.splat176 = shufflevector <8 x i32> %broadcast.splatinsert175, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert177 = insertelement <8 x i32> poison, i32 %12, i64 0
  %broadcast.splat178 = shufflevector <8 x i32> %broadcast.splatinsert177, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert179 = insertelement <8 x i32> poison, i32 %16, i64 0
  %broadcast.splat180 = shufflevector <8 x i32> %broadcast.splatinsert179, <8 x i32> poison, <8 x i32> zeroinitializer
  br label %vector.ph164

broadcast_add_fusion.clone.loop_header.dim.1.preheader.lr.ph.split.us.split.us.split.us.split.us.split.us.split.us: ; preds = %broadcast_add_fusion.clone.loop_header.dim.1.preheader.lr.ph.split.us.split.us.split.us.split.us.split.us
  br i1 %shft.chk19, label %broadcast_add_fusion.clone.loop_header.dim.1.preheader.lr.ph.split.us.split.us.split.us.split.us.split.us.split.us.split.us, label %vector.ph189.preheader

vector.ph189.preheader:                           ; preds = %broadcast_add_fusion.clone.loop_header.dim.1.preheader.lr.ph.split.us.split.us.split.us.split.us.split.us.split.us
  %broadcast.splatinsert190 = insertelement <8 x i32> poison, i32 %.fr, i64 0
  %broadcast.splat191 = shufflevector <8 x i32> %broadcast.splatinsert190, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert192 = insertelement <8 x i32> poison, i32 %3, i64 0
  %broadcast.splat193 = shufflevector <8 x i32> %broadcast.splatinsert192, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert194 = insertelement <8 x i32> poison, i32 %.fr4, i64 0
  %broadcast.splat195 = shufflevector <8 x i32> %broadcast.splatinsert194, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert196 = insertelement <8 x i32> poison, i32 %6, i64 0
  %broadcast.splat197 = shufflevector <8 x i32> %broadcast.splatinsert196, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert198 = insertelement <8 x i32> poison, i32 %.fr5, i64 0
  %broadcast.splat199 = shufflevector <8 x i32> %broadcast.splatinsert198, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert200 = insertelement <8 x i32> poison, i32 %9, i64 0
  %broadcast.splat201 = shufflevector <8 x i32> %broadcast.splatinsert200, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert202 = insertelement <8 x i32> poison, i32 %12, i64 0
  %broadcast.splat203 = shufflevector <8 x i32> %broadcast.splatinsert202, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert204 = insertelement <8 x i32> poison, i32 %16, i64 0
  %broadcast.splat205 = shufflevector <8 x i32> %broadcast.splatinsert204, <8 x i32> poison, <8 x i32> zeroinitializer
  br label %vector.ph189

broadcast_add_fusion.clone.loop_header.dim.1.preheader.lr.ph.split.us.split.us.split.us.split.us.split.us.split.us.split.us: ; preds = %broadcast_add_fusion.clone.loop_header.dim.1.preheader.lr.ph.split.us.split.us.split.us.split.us.split.us.split.us
  br i1 %shft.chk21, label %vector.ph239.preheader, label %vector.ph214.preheader

vector.ph214.preheader:                           ; preds = %broadcast_add_fusion.clone.loop_header.dim.1.preheader.lr.ph.split.us.split.us.split.us.split.us.split.us.split.us.split.us
  %broadcast.splatinsert215 = insertelement <8 x i32> poison, i32 %.fr, i64 0
  %broadcast.splat216 = shufflevector <8 x i32> %broadcast.splatinsert215, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert217 = insertelement <8 x i32> poison, i32 %3, i64 0
  %broadcast.splat218 = shufflevector <8 x i32> %broadcast.splatinsert217, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert219 = insertelement <8 x i32> poison, i32 %.fr4, i64 0
  %broadcast.splat220 = shufflevector <8 x i32> %broadcast.splatinsert219, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert221 = insertelement <8 x i32> poison, i32 %6, i64 0
  %broadcast.splat222 = shufflevector <8 x i32> %broadcast.splatinsert221, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert223 = insertelement <8 x i32> poison, i32 %.fr5, i64 0
  %broadcast.splat224 = shufflevector <8 x i32> %broadcast.splatinsert223, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert225 = insertelement <8 x i32> poison, i32 %9, i64 0
  %broadcast.splat226 = shufflevector <8 x i32> %broadcast.splatinsert225, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert227 = insertelement <8 x i32> poison, i32 %.fr6, i64 0
  %broadcast.splat228 = shufflevector <8 x i32> %broadcast.splatinsert227, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert229 = insertelement <8 x i32> poison, i32 %16, i64 0
  %broadcast.splat230 = shufflevector <8 x i32> %broadcast.splatinsert229, <8 x i32> poison, <8 x i32> zeroinitializer
  br label %vector.ph214

vector.ph239.preheader:                           ; preds = %broadcast_add_fusion.clone.loop_header.dim.1.preheader.lr.ph.split.us.split.us.split.us.split.us.split.us.split.us.split.us
  %broadcast.splatinsert240 = insertelement <8 x i32> poison, i32 %.fr, i64 0
  %broadcast.splat241 = shufflevector <8 x i32> %broadcast.splatinsert240, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert242 = insertelement <8 x i32> poison, i32 %3, i64 0
  %broadcast.splat243 = shufflevector <8 x i32> %broadcast.splatinsert242, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert244 = insertelement <8 x i32> poison, i32 %.fr4, i64 0
  %broadcast.splat245 = shufflevector <8 x i32> %broadcast.splatinsert244, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert246 = insertelement <8 x i32> poison, i32 %6, i64 0
  %broadcast.splat247 = shufflevector <8 x i32> %broadcast.splatinsert246, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert248 = insertelement <8 x i32> poison, i32 %.fr5, i64 0
  %broadcast.splat249 = shufflevector <8 x i32> %broadcast.splatinsert248, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert250 = insertelement <8 x i32> poison, i32 %9, i64 0
  %broadcast.splat251 = shufflevector <8 x i32> %broadcast.splatinsert250, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert252 = insertelement <8 x i32> poison, i32 %.fr6, i64 0
  %broadcast.splat253 = shufflevector <8 x i32> %broadcast.splatinsert252, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert254 = insertelement <8 x i32> poison, i32 %12, i64 0
  %broadcast.splat255 = shufflevector <8 x i32> %broadcast.splatinsert254, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert256 = insertelement <8 x i32> poison, i32 %16, i64 0
  %broadcast.splat257 = shufflevector <8 x i32> %broadcast.splatinsert256, <8 x i32> poison, <8 x i32> zeroinitializer
  br label %vector.ph239

vector.ph239:                                     ; preds = %vector.ph239.preheader, %broadcast_add_fusion.clone.loop_exit.dim.1.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us
  %broadcast_add_fusion.clone.invar_address.dim.0.03.us.us.us.us.us.us.us.us = phi i64 [ %invar.inc.us.us.us.us.us.us.us.us, %broadcast_add_fusion.clone.loop_exit.dim.1.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us ], [ %lo_dim_0, %vector.ph239.preheader ]
  br label %vector.body258

vector.body258:                                   ; preds = %vector.body258, %vector.ph239
  %index259 = phi i64 [ 0, %vector.ph239 ], [ %index.next262.1, %vector.body258 ]
  %17 = getelementptr inbounds [512 x [512 x i32]], ptr %arg3, i64 0, i64 %broadcast_add_fusion.clone.invar_address.dim.0.03.us.us.us.us.us.us.us.us, i64 %index259
  %wide.load260 = load <8 x i32>, ptr %17, align 64, !invariant.load !1, !noalias !6
  %18 = getelementptr inbounds [512 x [512 x i32]], ptr %arg2, i64 0, i64 %broadcast_add_fusion.clone.invar_address.dim.0.03.us.us.us.us.us.us.us.us, i64 %index259
  %wide.load261 = load <8 x i32>, ptr %18, align 64, !invariant.load !1, !noalias !6
  %19 = add <8 x i32> %wide.load261, %wide.load260
  %20 = shl <8 x i32> %wide.load261, %broadcast.splat241
  %21 = lshr <8 x i32> %wide.load261, %broadcast.splat243
  %22 = or disjoint <8 x i32> %21, %20
  %23 = xor <8 x i32> %22, %19
  %24 = add <8 x i32> %23, %19
  %25 = shl <8 x i32> %23, %broadcast.splat245
  %26 = lshr <8 x i32> %23, %broadcast.splat247
  %27 = or disjoint <8 x i32> %25, %26
  %28 = xor <8 x i32> %27, %24
  %29 = add <8 x i32> %28, %24
  %30 = shl <8 x i32> %28, %broadcast.splat249
  %31 = lshr <8 x i32> %28, %broadcast.splat251
  %32 = or disjoint <8 x i32> %30, %31
  %33 = xor <8 x i32> %32, %29
  %34 = add <8 x i32> %33, %29
  %35 = shl <8 x i32> %33, %broadcast.splat253
  %36 = lshr <8 x i32> %33, %broadcast.splat255
  %37 = or disjoint <8 x i32> %35, %36
  %38 = xor <8 x i32> %37, %34
  %39 = add <8 x i32> %broadcast.splat257, %38
  %40 = getelementptr inbounds [512 x [512 x i32]], ptr %arg5, i64 0, i64 %broadcast_add_fusion.clone.invar_address.dim.0.03.us.us.us.us.us.us.us.us, i64 %index259
  store <8 x i32> %39, ptr %40, align 64, !alias.scope !6
  %index.next262 = or disjoint i64 %index259, 8
  %41 = getelementptr inbounds [512 x [512 x i32]], ptr %arg3, i64 0, i64 %broadcast_add_fusion.clone.invar_address.dim.0.03.us.us.us.us.us.us.us.us, i64 %index.next262
  %wide.load260.1 = load <8 x i32>, ptr %41, align 32, !invariant.load !1, !noalias !6
  %42 = getelementptr inbounds [512 x [512 x i32]], ptr %arg2, i64 0, i64 %broadcast_add_fusion.clone.invar_address.dim.0.03.us.us.us.us.us.us.us.us, i64 %index.next262
  %wide.load261.1 = load <8 x i32>, ptr %42, align 32, !invariant.load !1, !noalias !6
  %43 = add <8 x i32> %wide.load261.1, %wide.load260.1
  %44 = shl <8 x i32> %wide.load261.1, %broadcast.splat241
  %45 = lshr <8 x i32> %wide.load261.1, %broadcast.splat243
  %46 = or disjoint <8 x i32> %45, %44
  %47 = xor <8 x i32> %46, %43
  %48 = add <8 x i32> %47, %43
  %49 = shl <8 x i32> %47, %broadcast.splat245
  %50 = lshr <8 x i32> %47, %broadcast.splat247
  %51 = or disjoint <8 x i32> %49, %50
  %52 = xor <8 x i32> %51, %48
  %53 = add <8 x i32> %52, %48
  %54 = shl <8 x i32> %52, %broadcast.splat249
  %55 = lshr <8 x i32> %52, %broadcast.splat251
  %56 = or disjoint <8 x i32> %54, %55
  %57 = xor <8 x i32> %56, %53
  %58 = add <8 x i32> %57, %53
  %59 = shl <8 x i32> %57, %broadcast.splat253
  %60 = lshr <8 x i32> %57, %broadcast.splat255
  %61 = or disjoint <8 x i32> %59, %60
  %62 = xor <8 x i32> %61, %58
  %63 = add <8 x i32> %broadcast.splat257, %62
  %64 = getelementptr inbounds [512 x [512 x i32]], ptr %arg5, i64 0, i64 %broadcast_add_fusion.clone.invar_address.dim.0.03.us.us.us.us.us.us.us.us, i64 %index.next262
  store <8 x i32> %63, ptr %64, align 32, !alias.scope !6
  %index.next262.1 = add nuw nsw i64 %index259, 16
  %65 = icmp eq i64 %index.next262.1, 512
  br i1 %65, label %broadcast_add_fusion.clone.loop_exit.dim.1.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us, label %vector.body258, !llvm.loop !9

broadcast_add_fusion.clone.loop_exit.dim.1.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us: ; preds = %vector.body258
  %invar.inc.us.us.us.us.us.us.us.us = add nuw nsw i64 %broadcast_add_fusion.clone.invar_address.dim.0.03.us.us.us.us.us.us.us.us, 1
  %exitcond31.not = icmp eq i64 %invar.inc.us.us.us.us.us.us.us.us, %up_dim_0
  br i1 %exitcond31.not, label %return, label %vector.ph239, !llvm.loop !12

vector.ph214:                                     ; preds = %vector.ph214.preheader, %broadcast_add_fusion.clone.loop_exit.dim.1.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us.split
  %broadcast_add_fusion.clone.invar_address.dim.0.03.us.us.us.us.us.us.us = phi i64 [ %invar.inc.us.us.us.us.us.us.us, %broadcast_add_fusion.clone.loop_exit.dim.1.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us.split ], [ %lo_dim_0, %vector.ph214.preheader ]
  br label %vector.body231

vector.body231:                                   ; preds = %vector.body231, %vector.ph214
  %index232 = phi i64 [ 0, %vector.ph214 ], [ %index.next235.1, %vector.body231 ]
  %66 = getelementptr inbounds [512 x [512 x i32]], ptr %arg3, i64 0, i64 %broadcast_add_fusion.clone.invar_address.dim.0.03.us.us.us.us.us.us.us, i64 %index232
  %wide.load233 = load <8 x i32>, ptr %66, align 64, !invariant.load !1, !noalias !6
  %67 = getelementptr inbounds [512 x [512 x i32]], ptr %arg2, i64 0, i64 %broadcast_add_fusion.clone.invar_address.dim.0.03.us.us.us.us.us.us.us, i64 %index232
  %wide.load234 = load <8 x i32>, ptr %67, align 64, !invariant.load !1, !noalias !6
  %68 = add <8 x i32> %wide.load234, %wide.load233
  %69 = shl <8 x i32> %wide.load234, %broadcast.splat216
  %70 = lshr <8 x i32> %wide.load234, %broadcast.splat218
  %71 = or disjoint <8 x i32> %70, %69
  %72 = xor <8 x i32> %71, %68
  %73 = add <8 x i32> %72, %68
  %74 = shl <8 x i32> %72, %broadcast.splat220
  %75 = lshr <8 x i32> %72, %broadcast.splat222
  %76 = or disjoint <8 x i32> %74, %75
  %77 = xor <8 x i32> %76, %73
  %78 = add <8 x i32> %77, %73
  %79 = shl <8 x i32> %77, %broadcast.splat224
  %80 = lshr <8 x i32> %77, %broadcast.splat226
  %81 = or disjoint <8 x i32> %79, %80
  %82 = xor <8 x i32> %81, %78
  %83 = add <8 x i32> %82, %78
  %84 = shl <8 x i32> %82, %broadcast.splat228
  %85 = xor <8 x i32> %84, %83
  %86 = add <8 x i32> %broadcast.splat230, %85
  %87 = getelementptr inbounds [512 x [512 x i32]], ptr %arg5, i64 0, i64 %broadcast_add_fusion.clone.invar_address.dim.0.03.us.us.us.us.us.us.us, i64 %index232
  store <8 x i32> %86, ptr %87, align 64, !alias.scope !6
  %index.next235 = or disjoint i64 %index232, 8
  %88 = getelementptr inbounds [512 x [512 x i32]], ptr %arg3, i64 0, i64 %broadcast_add_fusion.clone.invar_address.dim.0.03.us.us.us.us.us.us.us, i64 %index.next235
  %wide.load233.1 = load <8 x i32>, ptr %88, align 32, !invariant.load !1, !noalias !6
  %89 = getelementptr inbounds [512 x [512 x i32]], ptr %arg2, i64 0, i64 %broadcast_add_fusion.clone.invar_address.dim.0.03.us.us.us.us.us.us.us, i64 %index.next235
  %wide.load234.1 = load <8 x i32>, ptr %89, align 32, !invariant.load !1, !noalias !6
  %90 = add <8 x i32> %wide.load234.1, %wide.load233.1
  %91 = shl <8 x i32> %wide.load234.1, %broadcast.splat216
  %92 = lshr <8 x i32> %wide.load234.1, %broadcast.splat218
  %93 = or disjoint <8 x i32> %92, %91
  %94 = xor <8 x i32> %93, %90
  %95 = add <8 x i32> %94, %90
  %96 = shl <8 x i32> %94, %broadcast.splat220
  %97 = lshr <8 x i32> %94, %broadcast.splat222
  %98 = or disjoint <8 x i32> %96, %97
  %99 = xor <8 x i32> %98, %95
  %100 = add <8 x i32> %99, %95
  %101 = shl <8 x i32> %99, %broadcast.splat224
  %102 = lshr <8 x i32> %99, %broadcast.splat226
  %103 = or disjoint <8 x i32> %101, %102
  %104 = xor <8 x i32> %103, %100
  %105 = add <8 x i32> %104, %100
  %106 = shl <8 x i32> %104, %broadcast.splat228
  %107 = xor <8 x i32> %106, %105
  %108 = add <8 x i32> %broadcast.splat230, %107
  %109 = getelementptr inbounds [512 x [512 x i32]], ptr %arg5, i64 0, i64 %broadcast_add_fusion.clone.invar_address.dim.0.03.us.us.us.us.us.us.us, i64 %index.next235
  store <8 x i32> %108, ptr %109, align 32, !alias.scope !6
  %index.next235.1 = add nuw nsw i64 %index232, 16
  %110 = icmp eq i64 %index.next235.1, 512
  br i1 %110, label %broadcast_add_fusion.clone.loop_exit.dim.1.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us.split, label %vector.body231, !llvm.loop !14

broadcast_add_fusion.clone.loop_exit.dim.1.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us.split: ; preds = %vector.body231
  %invar.inc.us.us.us.us.us.us.us = add nuw nsw i64 %broadcast_add_fusion.clone.invar_address.dim.0.03.us.us.us.us.us.us.us, 1
  %exitcond29.not = icmp eq i64 %invar.inc.us.us.us.us.us.us.us, %up_dim_0
  br i1 %exitcond29.not, label %return, label %vector.ph214, !llvm.loop !12

vector.ph189:                                     ; preds = %vector.ph189.preheader, %broadcast_add_fusion.clone.loop_exit.dim.1.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us.split
  %broadcast_add_fusion.clone.invar_address.dim.0.03.us.us.us.us.us.us = phi i64 [ %invar.inc.us.us.us.us.us.us, %broadcast_add_fusion.clone.loop_exit.dim.1.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us.split ], [ %lo_dim_0, %vector.ph189.preheader ]
  br label %vector.body206

vector.body206:                                   ; preds = %vector.body206, %vector.ph189
  %index207 = phi i64 [ 0, %vector.ph189 ], [ %index.next210.1, %vector.body206 ]
  %111 = getelementptr inbounds [512 x [512 x i32]], ptr %arg3, i64 0, i64 %broadcast_add_fusion.clone.invar_address.dim.0.03.us.us.us.us.us.us, i64 %index207
  %wide.load208 = load <8 x i32>, ptr %111, align 64, !invariant.load !1, !noalias !6
  %112 = getelementptr inbounds [512 x [512 x i32]], ptr %arg2, i64 0, i64 %broadcast_add_fusion.clone.invar_address.dim.0.03.us.us.us.us.us.us, i64 %index207
  %wide.load209 = load <8 x i32>, ptr %112, align 64, !invariant.load !1, !noalias !6
  %113 = add <8 x i32> %wide.load209, %wide.load208
  %114 = shl <8 x i32> %wide.load209, %broadcast.splat191
  %115 = lshr <8 x i32> %wide.load209, %broadcast.splat193
  %116 = or disjoint <8 x i32> %115, %114
  %117 = xor <8 x i32> %116, %113
  %118 = add <8 x i32> %117, %113
  %119 = shl <8 x i32> %117, %broadcast.splat195
  %120 = lshr <8 x i32> %117, %broadcast.splat197
  %121 = or disjoint <8 x i32> %119, %120
  %122 = xor <8 x i32> %121, %118
  %123 = add <8 x i32> %122, %118
  %124 = shl <8 x i32> %122, %broadcast.splat199
  %125 = lshr <8 x i32> %122, %broadcast.splat201
  %126 = or disjoint <8 x i32> %124, %125
  %127 = xor <8 x i32> %126, %123
  %128 = add <8 x i32> %127, %123
  %129 = lshr <8 x i32> %127, %broadcast.splat203
  %130 = select i1 %shft.chk21, <8 x i32> %129, <8 x i32> zeroinitializer
  %131 = xor <8 x i32> %130, %128
  %132 = add <8 x i32> %broadcast.splat205, %131
  %133 = getelementptr inbounds [512 x [512 x i32]], ptr %arg5, i64 0, i64 %broadcast_add_fusion.clone.invar_address.dim.0.03.us.us.us.us.us.us, i64 %index207
  store <8 x i32> %132, ptr %133, align 64, !alias.scope !6
  %index.next210 = or disjoint i64 %index207, 8
  %134 = getelementptr inbounds [512 x [512 x i32]], ptr %arg3, i64 0, i64 %broadcast_add_fusion.clone.invar_address.dim.0.03.us.us.us.us.us.us, i64 %index.next210
  %wide.load208.1 = load <8 x i32>, ptr %134, align 32, !invariant.load !1, !noalias !6
  %135 = getelementptr inbounds [512 x [512 x i32]], ptr %arg2, i64 0, i64 %broadcast_add_fusion.clone.invar_address.dim.0.03.us.us.us.us.us.us, i64 %index.next210
  %wide.load209.1 = load <8 x i32>, ptr %135, align 32, !invariant.load !1, !noalias !6
  %136 = add <8 x i32> %wide.load209.1, %wide.load208.1
  %137 = shl <8 x i32> %wide.load209.1, %broadcast.splat191
  %138 = lshr <8 x i32> %wide.load209.1, %broadcast.splat193
  %139 = or disjoint <8 x i32> %138, %137
  %140 = xor <8 x i32> %139, %136
  %141 = add <8 x i32> %140, %136
  %142 = shl <8 x i32> %140, %broadcast.splat195
  %143 = lshr <8 x i32> %140, %broadcast.splat197
  %144 = or disjoint <8 x i32> %142, %143
  %145 = xor <8 x i32> %144, %141
  %146 = add <8 x i32> %145, %141
  %147 = shl <8 x i32> %145, %broadcast.splat199
  %148 = lshr <8 x i32> %145, %broadcast.splat201
  %149 = or disjoint <8 x i32> %147, %148
  %150 = xor <8 x i32> %149, %146
  %151 = add <8 x i32> %150, %146
  %152 = lshr <8 x i32> %150, %broadcast.splat203
  %153 = select i1 %shft.chk21, <8 x i32> %152, <8 x i32> zeroinitializer
  %154 = xor <8 x i32> %153, %151
  %155 = add <8 x i32> %broadcast.splat205, %154
  %156 = getelementptr inbounds [512 x [512 x i32]], ptr %arg5, i64 0, i64 %broadcast_add_fusion.clone.invar_address.dim.0.03.us.us.us.us.us.us, i64 %index.next210
  store <8 x i32> %155, ptr %156, align 32, !alias.scope !6
  %index.next210.1 = add nuw nsw i64 %index207, 16
  %157 = icmp eq i64 %index.next210.1, 512
  br i1 %157, label %broadcast_add_fusion.clone.loop_exit.dim.1.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us.split, label %vector.body206, !llvm.loop !15

broadcast_add_fusion.clone.loop_exit.dim.1.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us.split: ; preds = %vector.body206
  %invar.inc.us.us.us.us.us.us = add nuw nsw i64 %broadcast_add_fusion.clone.invar_address.dim.0.03.us.us.us.us.us.us, 1
  %exitcond27.not = icmp eq i64 %invar.inc.us.us.us.us.us.us, %up_dim_0
  br i1 %exitcond27.not, label %return, label %vector.ph189, !llvm.loop !12

vector.ph164:                                     ; preds = %vector.ph164.preheader, %broadcast_add_fusion.clone.loop_exit.dim.1.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us.split
  %broadcast_add_fusion.clone.invar_address.dim.0.03.us.us.us.us.us = phi i64 [ %invar.inc.us.us.us.us.us, %broadcast_add_fusion.clone.loop_exit.dim.1.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us.split ], [ %lo_dim_0, %vector.ph164.preheader ]
  br label %vector.body181

vector.body181:                                   ; preds = %vector.body181, %vector.ph164
  %index182 = phi i64 [ 0, %vector.ph164 ], [ %index.next185.1, %vector.body181 ]
  %158 = getelementptr inbounds [512 x [512 x i32]], ptr %arg3, i64 0, i64 %broadcast_add_fusion.clone.invar_address.dim.0.03.us.us.us.us.us, i64 %index182
  %wide.load183 = load <8 x i32>, ptr %158, align 64, !invariant.load !1, !noalias !6
  %159 = getelementptr inbounds [512 x [512 x i32]], ptr %arg2, i64 0, i64 %broadcast_add_fusion.clone.invar_address.dim.0.03.us.us.us.us.us, i64 %index182
  %wide.load184 = load <8 x i32>, ptr %159, align 64, !invariant.load !1, !noalias !6
  %160 = add <8 x i32> %wide.load184, %wide.load183
  %161 = shl <8 x i32> %wide.load184, %broadcast.splat166
  %162 = lshr <8 x i32> %wide.load184, %broadcast.splat168
  %163 = or disjoint <8 x i32> %162, %161
  %164 = xor <8 x i32> %163, %160
  %165 = add <8 x i32> %164, %160
  %166 = shl <8 x i32> %164, %broadcast.splat170
  %167 = lshr <8 x i32> %164, %broadcast.splat172
  %168 = or disjoint <8 x i32> %166, %167
  %169 = xor <8 x i32> %168, %165
  %170 = add <8 x i32> %169, %165
  %171 = shl <8 x i32> %169, %broadcast.splat174
  %172 = xor <8 x i32> %171, %170
  %173 = add <8 x i32> %172, %170
  %174 = shl <8 x i32> %172, %broadcast.splat176
  %175 = select i1 %shft.chk19, <8 x i32> %174, <8 x i32> zeroinitializer
  %176 = lshr <8 x i32> %172, %broadcast.splat178
  %177 = select i1 %shft.chk21, <8 x i32> %176, <8 x i32> zeroinitializer
  %178 = or <8 x i32> %175, %177
  %179 = xor <8 x i32> %178, %173
  %180 = add <8 x i32> %broadcast.splat180, %179
  %181 = getelementptr inbounds [512 x [512 x i32]], ptr %arg5, i64 0, i64 %broadcast_add_fusion.clone.invar_address.dim.0.03.us.us.us.us.us, i64 %index182
  store <8 x i32> %180, ptr %181, align 64, !alias.scope !6
  %index.next185 = or disjoint i64 %index182, 8
  %182 = getelementptr inbounds [512 x [512 x i32]], ptr %arg3, i64 0, i64 %broadcast_add_fusion.clone.invar_address.dim.0.03.us.us.us.us.us, i64 %index.next185
  %wide.load183.1 = load <8 x i32>, ptr %182, align 32, !invariant.load !1, !noalias !6
  %183 = getelementptr inbounds [512 x [512 x i32]], ptr %arg2, i64 0, i64 %broadcast_add_fusion.clone.invar_address.dim.0.03.us.us.us.us.us, i64 %index.next185
  %wide.load184.1 = load <8 x i32>, ptr %183, align 32, !invariant.load !1, !noalias !6
  %184 = add <8 x i32> %wide.load184.1, %wide.load183.1
  %185 = shl <8 x i32> %wide.load184.1, %broadcast.splat166
  %186 = lshr <8 x i32> %wide.load184.1, %broadcast.splat168
  %187 = or disjoint <8 x i32> %186, %185
  %188 = xor <8 x i32> %187, %184
  %189 = add <8 x i32> %188, %184
  %190 = shl <8 x i32> %188, %broadcast.splat170
  %191 = lshr <8 x i32> %188, %broadcast.splat172
  %192 = or disjoint <8 x i32> %190, %191
  %193 = xor <8 x i32> %192, %189
  %194 = add <8 x i32> %193, %189
  %195 = shl <8 x i32> %193, %broadcast.splat174
  %196 = xor <8 x i32> %195, %194
  %197 = add <8 x i32> %196, %194
  %198 = shl <8 x i32> %196, %broadcast.splat176
  %199 = select i1 %shft.chk19, <8 x i32> %198, <8 x i32> zeroinitializer
  %200 = lshr <8 x i32> %196, %broadcast.splat178
  %201 = select i1 %shft.chk21, <8 x i32> %200, <8 x i32> zeroinitializer
  %202 = or <8 x i32> %199, %201
  %203 = xor <8 x i32> %202, %197
  %204 = add <8 x i32> %broadcast.splat180, %203
  %205 = getelementptr inbounds [512 x [512 x i32]], ptr %arg5, i64 0, i64 %broadcast_add_fusion.clone.invar_address.dim.0.03.us.us.us.us.us, i64 %index.next185
  store <8 x i32> %204, ptr %205, align 32, !alias.scope !6
  %index.next185.1 = add nuw nsw i64 %index182, 16
  %206 = icmp eq i64 %index.next185.1, 512
  br i1 %206, label %broadcast_add_fusion.clone.loop_exit.dim.1.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us.split, label %vector.body181, !llvm.loop !16

broadcast_add_fusion.clone.loop_exit.dim.1.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us.split: ; preds = %vector.body181
  %invar.inc.us.us.us.us.us = add nuw nsw i64 %broadcast_add_fusion.clone.invar_address.dim.0.03.us.us.us.us.us, 1
  %exitcond25.not = icmp eq i64 %invar.inc.us.us.us.us.us, %up_dim_0
  br i1 %exitcond25.not, label %return, label %vector.ph164, !llvm.loop !12

vector.ph139:                                     ; preds = %vector.ph139.preheader, %broadcast_add_fusion.clone.loop_exit.dim.1.split.us.us.split.us.us.split.us.us.split.us.us.split
  %broadcast_add_fusion.clone.invar_address.dim.0.03.us.us.us.us = phi i64 [ %invar.inc.us.us.us.us, %broadcast_add_fusion.clone.loop_exit.dim.1.split.us.us.split.us.us.split.us.us.split.us.us.split ], [ %lo_dim_0, %vector.ph139.preheader ]
  br label %vector.body156

vector.body156:                                   ; preds = %vector.body156, %vector.ph139
  %index157 = phi i64 [ 0, %vector.ph139 ], [ %index.next160.1, %vector.body156 ]
  %207 = getelementptr inbounds [512 x [512 x i32]], ptr %arg3, i64 0, i64 %broadcast_add_fusion.clone.invar_address.dim.0.03.us.us.us.us, i64 %index157
  %wide.load158 = load <8 x i32>, ptr %207, align 64, !invariant.load !1, !noalias !6
  %208 = getelementptr inbounds [512 x [512 x i32]], ptr %arg2, i64 0, i64 %broadcast_add_fusion.clone.invar_address.dim.0.03.us.us.us.us, i64 %index157
  %wide.load159 = load <8 x i32>, ptr %208, align 64, !invariant.load !1, !noalias !6
  %209 = add <8 x i32> %wide.load159, %wide.load158
  %210 = shl <8 x i32> %wide.load159, %broadcast.splat141
  %211 = lshr <8 x i32> %wide.load159, %broadcast.splat143
  %212 = or disjoint <8 x i32> %211, %210
  %213 = xor <8 x i32> %212, %209
  %214 = add <8 x i32> %213, %209
  %215 = shl <8 x i32> %213, %broadcast.splat145
  %216 = lshr <8 x i32> %213, %broadcast.splat147
  %217 = or disjoint <8 x i32> %215, %216
  %218 = xor <8 x i32> %217, %214
  %219 = add <8 x i32> %218, %214
  %220 = lshr <8 x i32> %218, %broadcast.splat149
  %221 = select i1 %shft.chk18, <8 x i32> %220, <8 x i32> zeroinitializer
  %222 = xor <8 x i32> %221, %219
  %223 = add <8 x i32> %222, %219
  %224 = shl <8 x i32> %222, %broadcast.splat151
  %225 = select i1 %shft.chk19, <8 x i32> %224, <8 x i32> zeroinitializer
  %226 = lshr <8 x i32> %222, %broadcast.splat153
  %227 = select i1 %shft.chk21, <8 x i32> %226, <8 x i32> zeroinitializer
  %228 = or <8 x i32> %225, %227
  %229 = xor <8 x i32> %228, %223
  %230 = add <8 x i32> %broadcast.splat155, %229
  %231 = getelementptr inbounds [512 x [512 x i32]], ptr %arg5, i64 0, i64 %broadcast_add_fusion.clone.invar_address.dim.0.03.us.us.us.us, i64 %index157
  store <8 x i32> %230, ptr %231, align 64, !alias.scope !6
  %index.next160 = or disjoint i64 %index157, 8
  %232 = getelementptr inbounds [512 x [512 x i32]], ptr %arg3, i64 0, i64 %broadcast_add_fusion.clone.invar_address.dim.0.03.us.us.us.us, i64 %index.next160
  %wide.load158.1 = load <8 x i32>, ptr %232, align 32, !invariant.load !1, !noalias !6
  %233 = getelementptr inbounds [512 x [512 x i32]], ptr %arg2, i64 0, i64 %broadcast_add_fusion.clone.invar_address.dim.0.03.us.us.us.us, i64 %index.next160
  %wide.load159.1 = load <8 x i32>, ptr %233, align 32, !invariant.load !1, !noalias !6
  %234 = add <8 x i32> %wide.load159.1, %wide.load158.1
  %235 = shl <8 x i32> %wide.load159.1, %broadcast.splat141
  %236 = lshr <8 x i32> %wide.load159.1, %broadcast.splat143
  %237 = or disjoint <8 x i32> %236, %235
  %238 = xor <8 x i32> %237, %234
  %239 = add <8 x i32> %238, %234
  %240 = shl <8 x i32> %238, %broadcast.splat145
  %241 = lshr <8 x i32> %238, %broadcast.splat147
  %242 = or disjoint <8 x i32> %240, %241
  %243 = xor <8 x i32> %242, %239
  %244 = add <8 x i32> %243, %239
  %245 = lshr <8 x i32> %243, %broadcast.splat149
  %246 = select i1 %shft.chk18, <8 x i32> %245, <8 x i32> zeroinitializer
  %247 = xor <8 x i32> %246, %244
  %248 = add <8 x i32> %247, %244
  %249 = shl <8 x i32> %247, %broadcast.splat151
  %250 = select i1 %shft.chk19, <8 x i32> %249, <8 x i32> zeroinitializer
  %251 = lshr <8 x i32> %247, %broadcast.splat153
  %252 = select i1 %shft.chk21, <8 x i32> %251, <8 x i32> zeroinitializer
  %253 = or <8 x i32> %250, %252
  %254 = xor <8 x i32> %253, %248
  %255 = add <8 x i32> %broadcast.splat155, %254
  %256 = getelementptr inbounds [512 x [512 x i32]], ptr %arg5, i64 0, i64 %broadcast_add_fusion.clone.invar_address.dim.0.03.us.us.us.us, i64 %index.next160
  store <8 x i32> %255, ptr %256, align 32, !alias.scope !6
  %index.next160.1 = add nuw nsw i64 %index157, 16
  %257 = icmp eq i64 %index.next160.1, 512
  br i1 %257, label %broadcast_add_fusion.clone.loop_exit.dim.1.split.us.us.split.us.us.split.us.us.split.us.us.split, label %vector.body156, !llvm.loop !17

broadcast_add_fusion.clone.loop_exit.dim.1.split.us.us.split.us.us.split.us.us.split.us.us.split: ; preds = %vector.body156
  %invar.inc.us.us.us.us = add nuw nsw i64 %broadcast_add_fusion.clone.invar_address.dim.0.03.us.us.us.us, 1
  %exitcond23.not = icmp eq i64 %invar.inc.us.us.us.us, %up_dim_0
  br i1 %exitcond23.not, label %return, label %vector.ph139, !llvm.loop !12

vector.ph114:                                     ; preds = %vector.ph114.preheader, %broadcast_add_fusion.clone.loop_exit.dim.1.split.us.us.split.us.us.split.us.us.split
  %broadcast_add_fusion.clone.invar_address.dim.0.03.us.us.us = phi i64 [ %invar.inc.us.us.us, %broadcast_add_fusion.clone.loop_exit.dim.1.split.us.us.split.us.us.split.us.us.split ], [ %lo_dim_0, %vector.ph114.preheader ]
  br label %vector.body131

vector.body131:                                   ; preds = %vector.body131, %vector.ph114
  %index132 = phi i64 [ 0, %vector.ph114 ], [ %index.next135.1, %vector.body131 ]
  %258 = getelementptr inbounds [512 x [512 x i32]], ptr %arg3, i64 0, i64 %broadcast_add_fusion.clone.invar_address.dim.0.03.us.us.us, i64 %index132
  %wide.load133 = load <8 x i32>, ptr %258, align 64, !invariant.load !1, !noalias !6
  %259 = getelementptr inbounds [512 x [512 x i32]], ptr %arg2, i64 0, i64 %broadcast_add_fusion.clone.invar_address.dim.0.03.us.us.us, i64 %index132
  %wide.load134 = load <8 x i32>, ptr %259, align 64, !invariant.load !1, !noalias !6
  %260 = add <8 x i32> %wide.load134, %wide.load133
  %261 = shl <8 x i32> %wide.load134, %broadcast.splat116
  %262 = lshr <8 x i32> %wide.load134, %broadcast.splat118
  %263 = or disjoint <8 x i32> %262, %261
  %264 = xor <8 x i32> %263, %260
  %265 = add <8 x i32> %264, %260
  %266 = shl <8 x i32> %264, %broadcast.splat120
  %267 = xor <8 x i32> %266, %265
  %268 = add <8 x i32> %267, %265
  %269 = shl <8 x i32> %267, %broadcast.splat122
  %270 = select i1 %shft.chk16, <8 x i32> %269, <8 x i32> zeroinitializer
  %271 = lshr <8 x i32> %267, %broadcast.splat124
  %272 = select i1 %shft.chk18, <8 x i32> %271, <8 x i32> zeroinitializer
  %273 = or <8 x i32> %270, %272
  %274 = xor <8 x i32> %273, %268
  %275 = add <8 x i32> %274, %268
  %276 = shl <8 x i32> %274, %broadcast.splat126
  %277 = select i1 %shft.chk19, <8 x i32> %276, <8 x i32> zeroinitializer
  %278 = lshr <8 x i32> %274, %broadcast.splat128
  %279 = select i1 %shft.chk21, <8 x i32> %278, <8 x i32> zeroinitializer
  %280 = or <8 x i32> %277, %279
  %281 = xor <8 x i32> %280, %275
  %282 = add <8 x i32> %broadcast.splat130, %281
  %283 = getelementptr inbounds [512 x [512 x i32]], ptr %arg5, i64 0, i64 %broadcast_add_fusion.clone.invar_address.dim.0.03.us.us.us, i64 %index132
  store <8 x i32> %282, ptr %283, align 64, !alias.scope !6
  %index.next135 = or disjoint i64 %index132, 8
  %284 = getelementptr inbounds [512 x [512 x i32]], ptr %arg3, i64 0, i64 %broadcast_add_fusion.clone.invar_address.dim.0.03.us.us.us, i64 %index.next135
  %wide.load133.1 = load <8 x i32>, ptr %284, align 32, !invariant.load !1, !noalias !6
  %285 = getelementptr inbounds [512 x [512 x i32]], ptr %arg2, i64 0, i64 %broadcast_add_fusion.clone.invar_address.dim.0.03.us.us.us, i64 %index.next135
  %wide.load134.1 = load <8 x i32>, ptr %285, align 32, !invariant.load !1, !noalias !6
  %286 = add <8 x i32> %wide.load134.1, %wide.load133.1
  %287 = shl <8 x i32> %wide.load134.1, %broadcast.splat116
  %288 = lshr <8 x i32> %wide.load134.1, %broadcast.splat118
  %289 = or disjoint <8 x i32> %288, %287
  %290 = xor <8 x i32> %289, %286
  %291 = add <8 x i32> %290, %286
  %292 = shl <8 x i32> %290, %broadcast.splat120
  %293 = xor <8 x i32> %292, %291
  %294 = add <8 x i32> %293, %291
  %295 = shl <8 x i32> %293, %broadcast.splat122
  %296 = select i1 %shft.chk16, <8 x i32> %295, <8 x i32> zeroinitializer
  %297 = lshr <8 x i32> %293, %broadcast.splat124
  %298 = select i1 %shft.chk18, <8 x i32> %297, <8 x i32> zeroinitializer
  %299 = or <8 x i32> %296, %298
  %300 = xor <8 x i32> %299, %294
  %301 = add <8 x i32> %300, %294
  %302 = shl <8 x i32> %300, %broadcast.splat126
  %303 = select i1 %shft.chk19, <8 x i32> %302, <8 x i32> zeroinitializer
  %304 = lshr <8 x i32> %300, %broadcast.splat128
  %305 = select i1 %shft.chk21, <8 x i32> %304, <8 x i32> zeroinitializer
  %306 = or <8 x i32> %303, %305
  %307 = xor <8 x i32> %306, %301
  %308 = add <8 x i32> %broadcast.splat130, %307
  %309 = getelementptr inbounds [512 x [512 x i32]], ptr %arg5, i64 0, i64 %broadcast_add_fusion.clone.invar_address.dim.0.03.us.us.us, i64 %index.next135
  store <8 x i32> %308, ptr %309, align 32, !alias.scope !6
  %index.next135.1 = add nuw nsw i64 %index132, 16
  %310 = icmp eq i64 %index.next135.1, 512
  br i1 %310, label %broadcast_add_fusion.clone.loop_exit.dim.1.split.us.us.split.us.us.split.us.us.split, label %vector.body131, !llvm.loop !18

broadcast_add_fusion.clone.loop_exit.dim.1.split.us.us.split.us.us.split.us.us.split: ; preds = %vector.body131
  %invar.inc.us.us.us = add nuw nsw i64 %broadcast_add_fusion.clone.invar_address.dim.0.03.us.us.us, 1
  %exitcond21.not = icmp eq i64 %invar.inc.us.us.us, %up_dim_0
  br i1 %exitcond21.not, label %return, label %vector.ph114, !llvm.loop !12

vector.ph89:                                      ; preds = %vector.ph89.preheader, %broadcast_add_fusion.clone.loop_exit.dim.1.split.us.us.split.us.us.split
  %broadcast_add_fusion.clone.invar_address.dim.0.03.us.us = phi i64 [ %invar.inc.us.us, %broadcast_add_fusion.clone.loop_exit.dim.1.split.us.us.split.us.us.split ], [ %lo_dim_0, %vector.ph89.preheader ]
  br label %vector.body106

vector.body106:                                   ; preds = %vector.body106, %vector.ph89
  %index107 = phi i64 [ 0, %vector.ph89 ], [ %index.next110.1, %vector.body106 ]
  %311 = getelementptr inbounds [512 x [512 x i32]], ptr %arg3, i64 0, i64 %broadcast_add_fusion.clone.invar_address.dim.0.03.us.us, i64 %index107
  %wide.load108 = load <8 x i32>, ptr %311, align 64, !invariant.load !1, !noalias !6
  %312 = getelementptr inbounds [512 x [512 x i32]], ptr %arg2, i64 0, i64 %broadcast_add_fusion.clone.invar_address.dim.0.03.us.us, i64 %index107
  %wide.load109 = load <8 x i32>, ptr %312, align 64, !invariant.load !1, !noalias !6
  %313 = add <8 x i32> %wide.load109, %wide.load108
  %314 = shl <8 x i32> %wide.load109, %broadcast.splat91
  %315 = lshr <8 x i32> %wide.load109, %broadcast.splat93
  %316 = or disjoint <8 x i32> %315, %314
  %317 = xor <8 x i32> %316, %313
  %318 = add <8 x i32> %317, %313
  %319 = lshr <8 x i32> %317, %broadcast.splat95
  %320 = select i1 %shft.chk15, <8 x i32> %319, <8 x i32> zeroinitializer
  %321 = xor <8 x i32> %320, %318
  %322 = add <8 x i32> %321, %318
  %323 = shl <8 x i32> %321, %broadcast.splat97
  %324 = select i1 %shft.chk16, <8 x i32> %323, <8 x i32> zeroinitializer
  %325 = lshr <8 x i32> %321, %broadcast.splat99
  %326 = select i1 %shft.chk18, <8 x i32> %325, <8 x i32> zeroinitializer
  %327 = or <8 x i32> %324, %326
  %328 = xor <8 x i32> %327, %322
  %329 = add <8 x i32> %328, %322
  %330 = shl <8 x i32> %328, %broadcast.splat101
  %331 = select i1 %shft.chk19, <8 x i32> %330, <8 x i32> zeroinitializer
  %332 = lshr <8 x i32> %328, %broadcast.splat103
  %333 = select i1 %shft.chk21, <8 x i32> %332, <8 x i32> zeroinitializer
  %334 = or <8 x i32> %331, %333
  %335 = xor <8 x i32> %334, %329
  %336 = add <8 x i32> %broadcast.splat105, %335
  %337 = getelementptr inbounds [512 x [512 x i32]], ptr %arg5, i64 0, i64 %broadcast_add_fusion.clone.invar_address.dim.0.03.us.us, i64 %index107
  store <8 x i32> %336, ptr %337, align 64, !alias.scope !6
  %index.next110 = or disjoint i64 %index107, 8
  %338 = getelementptr inbounds [512 x [512 x i32]], ptr %arg3, i64 0, i64 %broadcast_add_fusion.clone.invar_address.dim.0.03.us.us, i64 %index.next110
  %wide.load108.1 = load <8 x i32>, ptr %338, align 32, !invariant.load !1, !noalias !6
  %339 = getelementptr inbounds [512 x [512 x i32]], ptr %arg2, i64 0, i64 %broadcast_add_fusion.clone.invar_address.dim.0.03.us.us, i64 %index.next110
  %wide.load109.1 = load <8 x i32>, ptr %339, align 32, !invariant.load !1, !noalias !6
  %340 = add <8 x i32> %wide.load109.1, %wide.load108.1
  %341 = shl <8 x i32> %wide.load109.1, %broadcast.splat91
  %342 = lshr <8 x i32> %wide.load109.1, %broadcast.splat93
  %343 = or disjoint <8 x i32> %342, %341
  %344 = xor <8 x i32> %343, %340
  %345 = add <8 x i32> %344, %340
  %346 = lshr <8 x i32> %344, %broadcast.splat95
  %347 = select i1 %shft.chk15, <8 x i32> %346, <8 x i32> zeroinitializer
  %348 = xor <8 x i32> %347, %345
  %349 = add <8 x i32> %348, %345
  %350 = shl <8 x i32> %348, %broadcast.splat97
  %351 = select i1 %shft.chk16, <8 x i32> %350, <8 x i32> zeroinitializer
  %352 = lshr <8 x i32> %348, %broadcast.splat99
  %353 = select i1 %shft.chk18, <8 x i32> %352, <8 x i32> zeroinitializer
  %354 = or <8 x i32> %351, %353
  %355 = xor <8 x i32> %354, %349
  %356 = add <8 x i32> %355, %349
  %357 = shl <8 x i32> %355, %broadcast.splat101
  %358 = select i1 %shft.chk19, <8 x i32> %357, <8 x i32> zeroinitializer
  %359 = lshr <8 x i32> %355, %broadcast.splat103
  %360 = select i1 %shft.chk21, <8 x i32> %359, <8 x i32> zeroinitializer
  %361 = or <8 x i32> %358, %360
  %362 = xor <8 x i32> %361, %356
  %363 = add <8 x i32> %broadcast.splat105, %362
  %364 = getelementptr inbounds [512 x [512 x i32]], ptr %arg5, i64 0, i64 %broadcast_add_fusion.clone.invar_address.dim.0.03.us.us, i64 %index.next110
  store <8 x i32> %363, ptr %364, align 32, !alias.scope !6
  %index.next110.1 = add nuw nsw i64 %index107, 16
  %365 = icmp eq i64 %index.next110.1, 512
  br i1 %365, label %broadcast_add_fusion.clone.loop_exit.dim.1.split.us.us.split.us.us.split, label %vector.body106, !llvm.loop !19

broadcast_add_fusion.clone.loop_exit.dim.1.split.us.us.split.us.us.split: ; preds = %vector.body106
  %invar.inc.us.us = add nuw nsw i64 %broadcast_add_fusion.clone.invar_address.dim.0.03.us.us, 1
  %exitcond19.not = icmp eq i64 %invar.inc.us.us, %up_dim_0
  br i1 %exitcond19.not, label %return, label %vector.ph89, !llvm.loop !12

vector.ph64:                                      ; preds = %vector.ph64.preheader, %broadcast_add_fusion.clone.loop_exit.dim.1.split.us.us.split
  %broadcast_add_fusion.clone.invar_address.dim.0.03.us = phi i64 [ %invar.inc.us, %broadcast_add_fusion.clone.loop_exit.dim.1.split.us.us.split ], [ %lo_dim_0, %vector.ph64.preheader ]
  br label %vector.body81

vector.body81:                                    ; preds = %vector.body81, %vector.ph64
  %index82 = phi i64 [ 0, %vector.ph64 ], [ %index.next85.1, %vector.body81 ]
  %366 = getelementptr inbounds [512 x [512 x i32]], ptr %arg3, i64 0, i64 %broadcast_add_fusion.clone.invar_address.dim.0.03.us, i64 %index82
  %wide.load83 = load <8 x i32>, ptr %366, align 64, !invariant.load !1, !noalias !6
  %367 = getelementptr inbounds [512 x [512 x i32]], ptr %arg2, i64 0, i64 %broadcast_add_fusion.clone.invar_address.dim.0.03.us, i64 %index82
  %wide.load84 = load <8 x i32>, ptr %367, align 64, !invariant.load !1, !noalias !6
  %368 = add <8 x i32> %wide.load84, %wide.load83
  %369 = shl <8 x i32> %wide.load84, %broadcast.splat66
  %370 = xor <8 x i32> %369, %368
  %371 = add <8 x i32> %370, %368
  %372 = shl <8 x i32> %370, %broadcast.splat68
  %373 = select i1 %shft.chk13, <8 x i32> %372, <8 x i32> zeroinitializer
  %374 = lshr <8 x i32> %370, %broadcast.splat70
  %375 = select i1 %shft.chk15, <8 x i32> %374, <8 x i32> zeroinitializer
  %376 = or <8 x i32> %373, %375
  %377 = xor <8 x i32> %376, %371
  %378 = add <8 x i32> %377, %371
  %379 = shl <8 x i32> %377, %broadcast.splat72
  %380 = select i1 %shft.chk16, <8 x i32> %379, <8 x i32> zeroinitializer
  %381 = lshr <8 x i32> %377, %broadcast.splat74
  %382 = select i1 %shft.chk18, <8 x i32> %381, <8 x i32> zeroinitializer
  %383 = or <8 x i32> %380, %382
  %384 = xor <8 x i32> %383, %378
  %385 = add <8 x i32> %384, %378
  %386 = shl <8 x i32> %384, %broadcast.splat76
  %387 = select i1 %shft.chk19, <8 x i32> %386, <8 x i32> zeroinitializer
  %388 = lshr <8 x i32> %384, %broadcast.splat78
  %389 = select i1 %shft.chk21, <8 x i32> %388, <8 x i32> zeroinitializer
  %390 = or <8 x i32> %387, %389
  %391 = xor <8 x i32> %390, %385
  %392 = add <8 x i32> %broadcast.splat80, %391
  %393 = getelementptr inbounds [512 x [512 x i32]], ptr %arg5, i64 0, i64 %broadcast_add_fusion.clone.invar_address.dim.0.03.us, i64 %index82
  store <8 x i32> %392, ptr %393, align 64, !alias.scope !6
  %index.next85 = or disjoint i64 %index82, 8
  %394 = getelementptr inbounds [512 x [512 x i32]], ptr %arg3, i64 0, i64 %broadcast_add_fusion.clone.invar_address.dim.0.03.us, i64 %index.next85
  %wide.load83.1 = load <8 x i32>, ptr %394, align 32, !invariant.load !1, !noalias !6
  %395 = getelementptr inbounds [512 x [512 x i32]], ptr %arg2, i64 0, i64 %broadcast_add_fusion.clone.invar_address.dim.0.03.us, i64 %index.next85
  %wide.load84.1 = load <8 x i32>, ptr %395, align 32, !invariant.load !1, !noalias !6
  %396 = add <8 x i32> %wide.load84.1, %wide.load83.1
  %397 = shl <8 x i32> %wide.load84.1, %broadcast.splat66
  %398 = xor <8 x i32> %397, %396
  %399 = add <8 x i32> %398, %396
  %400 = shl <8 x i32> %398, %broadcast.splat68
  %401 = select i1 %shft.chk13, <8 x i32> %400, <8 x i32> zeroinitializer
  %402 = lshr <8 x i32> %398, %broadcast.splat70
  %403 = select i1 %shft.chk15, <8 x i32> %402, <8 x i32> zeroinitializer
  %404 = or <8 x i32> %401, %403
  %405 = xor <8 x i32> %404, %399
  %406 = add <8 x i32> %405, %399
  %407 = shl <8 x i32> %405, %broadcast.splat72
  %408 = select i1 %shft.chk16, <8 x i32> %407, <8 x i32> zeroinitializer
  %409 = lshr <8 x i32> %405, %broadcast.splat74
  %410 = select i1 %shft.chk18, <8 x i32> %409, <8 x i32> zeroinitializer
  %411 = or <8 x i32> %408, %410
  %412 = xor <8 x i32> %411, %406
  %413 = add <8 x i32> %412, %406
  %414 = shl <8 x i32> %412, %broadcast.splat76
  %415 = select i1 %shft.chk19, <8 x i32> %414, <8 x i32> zeroinitializer
  %416 = lshr <8 x i32> %412, %broadcast.splat78
  %417 = select i1 %shft.chk21, <8 x i32> %416, <8 x i32> zeroinitializer
  %418 = or <8 x i32> %415, %417
  %419 = xor <8 x i32> %418, %413
  %420 = add <8 x i32> %broadcast.splat80, %419
  %421 = getelementptr inbounds [512 x [512 x i32]], ptr %arg5, i64 0, i64 %broadcast_add_fusion.clone.invar_address.dim.0.03.us, i64 %index.next85
  store <8 x i32> %420, ptr %421, align 32, !alias.scope !6
  %index.next85.1 = add nuw nsw i64 %index82, 16
  %422 = icmp eq i64 %index.next85.1, 512
  br i1 %422, label %broadcast_add_fusion.clone.loop_exit.dim.1.split.us.us.split, label %vector.body81, !llvm.loop !20

broadcast_add_fusion.clone.loop_exit.dim.1.split.us.us.split: ; preds = %vector.body81
  %invar.inc.us = add nuw nsw i64 %broadcast_add_fusion.clone.invar_address.dim.0.03.us, 1
  %exitcond17.not = icmp eq i64 %invar.inc.us, %up_dim_0
  br i1 %exitcond17.not, label %return, label %vector.ph64, !llvm.loop !12

vector.ph:                                        ; preds = %vector.ph.preheader, %broadcast_add_fusion.clone.loop_exit.dim.1.split
  %broadcast_add_fusion.clone.invar_address.dim.0.03 = phi i64 [ %invar.inc, %broadcast_add_fusion.clone.loop_exit.dim.1.split ], [ %lo_dim_0, %vector.ph.preheader ]
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next.1, %vector.body ]
  %423 = getelementptr inbounds [512 x [512 x i32]], ptr %arg3, i64 0, i64 %broadcast_add_fusion.clone.invar_address.dim.0.03, i64 %index
  %wide.load = load <8 x i32>, ptr %423, align 64, !invariant.load !1, !noalias !6
  %424 = getelementptr inbounds [512 x [512 x i32]], ptr %arg2, i64 0, i64 %broadcast_add_fusion.clone.invar_address.dim.0.03, i64 %index
  %wide.load62 = load <8 x i32>, ptr %424, align 64, !invariant.load !1, !noalias !6
  %425 = add <8 x i32> %wide.load62, %wide.load
  %426 = lshr <8 x i32> %wide.load62, %broadcast.splat
  %427 = select i1 %shft.chk12, <8 x i32> %426, <8 x i32> zeroinitializer
  %428 = xor <8 x i32> %427, %425
  %429 = add <8 x i32> %428, %425
  %430 = shl <8 x i32> %428, %broadcast.splat49
  %431 = select i1 %shft.chk13, <8 x i32> %430, <8 x i32> zeroinitializer
  %432 = lshr <8 x i32> %428, %broadcast.splat51
  %433 = select i1 %shft.chk15, <8 x i32> %432, <8 x i32> zeroinitializer
  %434 = or <8 x i32> %431, %433
  %435 = xor <8 x i32> %434, %429
  %436 = add <8 x i32> %435, %429
  %437 = shl <8 x i32> %435, %broadcast.splat53
  %438 = select i1 %shft.chk16, <8 x i32> %437, <8 x i32> zeroinitializer
  %439 = lshr <8 x i32> %435, %broadcast.splat55
  %440 = select i1 %shft.chk18, <8 x i32> %439, <8 x i32> zeroinitializer
  %441 = or <8 x i32> %438, %440
  %442 = xor <8 x i32> %441, %436
  %443 = add <8 x i32> %442, %436
  %444 = shl <8 x i32> %442, %broadcast.splat57
  %445 = select i1 %shft.chk19, <8 x i32> %444, <8 x i32> zeroinitializer
  %446 = lshr <8 x i32> %442, %broadcast.splat59
  %447 = select i1 %shft.chk21, <8 x i32> %446, <8 x i32> zeroinitializer
  %448 = or <8 x i32> %445, %447
  %449 = xor <8 x i32> %448, %443
  %450 = add <8 x i32> %broadcast.splat61, %449
  %451 = getelementptr inbounds [512 x [512 x i32]], ptr %arg5, i64 0, i64 %broadcast_add_fusion.clone.invar_address.dim.0.03, i64 %index
  store <8 x i32> %450, ptr %451, align 64, !alias.scope !6
  %index.next = or disjoint i64 %index, 8
  %452 = getelementptr inbounds [512 x [512 x i32]], ptr %arg3, i64 0, i64 %broadcast_add_fusion.clone.invar_address.dim.0.03, i64 %index.next
  %wide.load.1 = load <8 x i32>, ptr %452, align 32, !invariant.load !1, !noalias !6
  %453 = getelementptr inbounds [512 x [512 x i32]], ptr %arg2, i64 0, i64 %broadcast_add_fusion.clone.invar_address.dim.0.03, i64 %index.next
  %wide.load62.1 = load <8 x i32>, ptr %453, align 32, !invariant.load !1, !noalias !6
  %454 = add <8 x i32> %wide.load62.1, %wide.load.1
  %455 = lshr <8 x i32> %wide.load62.1, %broadcast.splat
  %456 = select i1 %shft.chk12, <8 x i32> %455, <8 x i32> zeroinitializer
  %457 = xor <8 x i32> %456, %454
  %458 = add <8 x i32> %457, %454
  %459 = shl <8 x i32> %457, %broadcast.splat49
  %460 = select i1 %shft.chk13, <8 x i32> %459, <8 x i32> zeroinitializer
  %461 = lshr <8 x i32> %457, %broadcast.splat51
  %462 = select i1 %shft.chk15, <8 x i32> %461, <8 x i32> zeroinitializer
  %463 = or <8 x i32> %460, %462
  %464 = xor <8 x i32> %463, %458
  %465 = add <8 x i32> %464, %458
  %466 = shl <8 x i32> %464, %broadcast.splat53
  %467 = select i1 %shft.chk16, <8 x i32> %466, <8 x i32> zeroinitializer
  %468 = lshr <8 x i32> %464, %broadcast.splat55
  %469 = select i1 %shft.chk18, <8 x i32> %468, <8 x i32> zeroinitializer
  %470 = or <8 x i32> %467, %469
  %471 = xor <8 x i32> %470, %465
  %472 = add <8 x i32> %471, %465
  %473 = shl <8 x i32> %471, %broadcast.splat57
  %474 = select i1 %shft.chk19, <8 x i32> %473, <8 x i32> zeroinitializer
  %475 = lshr <8 x i32> %471, %broadcast.splat59
  %476 = select i1 %shft.chk21, <8 x i32> %475, <8 x i32> zeroinitializer
  %477 = or <8 x i32> %474, %476
  %478 = xor <8 x i32> %477, %472
  %479 = add <8 x i32> %broadcast.splat61, %478
  %480 = getelementptr inbounds [512 x [512 x i32]], ptr %arg5, i64 0, i64 %broadcast_add_fusion.clone.invar_address.dim.0.03, i64 %index.next
  store <8 x i32> %479, ptr %480, align 32, !alias.scope !6
  %index.next.1 = add nuw nsw i64 %index, 16
  %481 = icmp eq i64 %index.next.1, 512
  br i1 %481, label %broadcast_add_fusion.clone.loop_exit.dim.1.split, label %vector.body, !llvm.loop !21

broadcast_add_fusion.clone.loop_exit.dim.1.split: ; preds = %vector.body
  %invar.inc = add nuw nsw i64 %broadcast_add_fusion.clone.invar_address.dim.0.03, 1
  %exitcond15.not = icmp eq i64 %invar.inc, %up_dim_0
  br i1 %exitcond15.not, label %return, label %vector.ph, !llvm.loop !12

return:                                           ; preds = %broadcast_add_fusion.clone.loop_exit.dim.1.split, %broadcast_add_fusion.clone.loop_exit.dim.1.split.us.us.split, %broadcast_add_fusion.clone.loop_exit.dim.1.split.us.us.split.us.us.split, %broadcast_add_fusion.clone.loop_exit.dim.1.split.us.us.split.us.us.split.us.us.split, %broadcast_add_fusion.clone.loop_exit.dim.1.split.us.us.split.us.us.split.us.us.split.us.us.split, %broadcast_add_fusion.clone.loop_exit.dim.1.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us.split, %broadcast_add_fusion.clone.loop_exit.dim.1.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us.split, %broadcast_add_fusion.clone.loop_exit.dim.1.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us.split, %broadcast_add_fusion.clone.loop_exit.dim.1.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us, %1
  ret ptr null
}

attributes #0 = { nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable "frame-pointer"="all" "prefer-vector-width"="256" }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"xla_dylib_index", i64 4}
!1 = !{}
!2 = !{i64 1048576}
!3 = !{i64 64}
!4 = !{i64 4}
!5 = !{i64 16}
!6 = !{!7}
!7 = !{!"result slice: {index:7, offset:2097216, size:1048576}", !8}
!8 = !{!"XLA host kernel broadcast_add_fusion.clone AA domain"}
!9 = distinct !{!9, !10, !11}
!10 = !{!"llvm.loop.isvectorized", i32 1}
!11 = !{!"llvm.loop.unroll.runtime.disable"}
!12 = distinct !{!12, !13}
!13 = !{!"llvm.loop.unroll.disable"}
!14 = distinct !{!14, !10, !11}
!15 = distinct !{!15, !10, !11}
!16 = distinct !{!16, !10, !11}
!17 = distinct !{!17, !10, !11}
!18 = distinct !{!18, !10, !11}
!19 = distinct !{!19, !10, !11}
!20 = distinct !{!20, !10, !11}
!21 = distinct !{!21, !10, !11}

; ModuleID = '__compute_module_part_05'
source_filename = "__compute_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define noalias noundef ptr @bitcast_concatenate_fusion(ptr readonly captures(none) %0) local_unnamed_addr #0 {
bitcast_concatenate_fusion.loop_header.dim.1.preheader:
  %args_gep = getelementptr inbounds nuw i8, ptr %0, i64 24
  %args = load ptr, ptr %args_gep, align 8
  %arg0 = load ptr, ptr %args, align 8, !invariant.load !1, !dereferenceable !2, !align !3
  %arg1_gep = getelementptr i8, ptr %args, i64 16
  %arg1 = load ptr, ptr %arg1_gep, align 8, !invariant.load !1, !dereferenceable !2, !align !3
  %arg2_gep = getelementptr i8, ptr %args, i64 32
  %arg2 = load ptr, ptr %arg2_gep, align 8, !invariant.load !1, !dereferenceable !4, !align !3
  %1 = load i32, ptr %arg1, align 64, !invariant.load !1, !noalias !5
  store i32 %1, ptr %arg2, align 64, !alias.scope !5
  %2 = load i32, ptr %arg0, align 64, !invariant.load !1, !noalias !5
  %3 = getelementptr inbounds nuw i8, ptr %arg2, i64 4
  store i32 %2, ptr %3, align 4, !alias.scope !5
  %.in.c = getelementptr inbounds nuw i8, ptr %arg1, i64 4
  %4 = load i32, ptr %.in.c, align 4, !invariant.load !1, !noalias !5
  %5 = getelementptr inbounds nuw i8, ptr %arg2, i64 8
  store i32 %4, ptr %5, align 8, !alias.scope !5
  %.in.1.c = getelementptr inbounds nuw i8, ptr %arg0, i64 4
  %6 = load i32, ptr %.in.1.c, align 4, !invariant.load !1, !noalias !5
  %7 = getelementptr inbounds nuw i8, ptr %arg2, i64 12
  store i32 %6, ptr %7, align 4, !alias.scope !5
  ret ptr null
}

attributes #0 = { nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable "frame-pointer"="all" "prefer-vector-width"="256" }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"xla_dylib_index", i64 5}
!1 = !{}
!2 = !{i64 8}
!3 = !{i64 64}
!4 = !{i64 16}
!5 = !{!6}
!6 = !{!"result slice: {index:1, offset:0, size:16}", !7}
!7 = !{!"XLA host kernel bitcast_concatenate_fusion AA domain"}
